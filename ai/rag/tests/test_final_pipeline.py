"""다운로드나 외부 LLM 없이 최종 파이프라인 계약을 검증한다."""

from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ai.llm.evaluation.retrieval_eval import evaluate, validate_no_relic_labels_in_queries
from ai.llm.prompts import (
    DOCENT_SYSTEM_PROMPT,
    FIRST_FOLLOW_UP_INSTRUCTION,
    INITIAL_DESCRIPTION_INSTRUCTION,
    LATER_FOLLOW_UP_INSTRUCTION,
    VISITOR_INSTRUCTIONS,
)
from ai.rag.chat import DocentChatService
from ai.rag.config import load_config, project_path
from ai.rag.document_builder import build_documents
from ai.rag.gold_loader import get_relic_records_by_label, gold_summary, load_gold
from ai.rag.retriever import retrieve
from ai.rag.vector_store import build_chroma, close_chroma, open_chroma


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(map(ord, text)) % 97)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 97)]


class FakeStore:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int):
        self.queries.append((query, k))
        return [(document, index / 10) for index, document in enumerate(self.documents[:k])]


class FinalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("ai/llm/configs/base.yaml")
        cls.gold_path = project_path(cls.config["data"]["gold_path"])
        cls.rows = load_gold(cls.gold_path)

    def test_gold_schema_and_existing_ids_are_preserved(self):
        summary = gold_summary(self.rows)
        self.assertEqual(summary["records"], 4269)
        self.assertEqual(summary["unique_chunk_ids"], 4269)
        self.assertEqual(summary["empty_documents"], 0)
        documents = build_documents(self.rows)
        self.assertEqual(len(documents), len(self.rows))
        self.assertEqual([document.id for document in documents], [row["chunk_id"] for row in self.rows])
        self.assertIn("유물 정보:", documents[0].page_content)
        self.assertNotIn(str(self.rows[0]), documents[0].page_content)

    def test_direct_lookup_returns_all_existing_chunks_for_label(self):
        records = get_relic_records_by_label(self.gold_path, "본관 1958")
        self.assertEqual(
            {record["chunk_id"] for record in records},
            {"nmm-bon-001958-00:profile", "nmm-bon-001958-00:desc-00"},
        )

    def test_chroma_one_document_per_gold_record_and_persist_reload(self):
        documents = build_documents(self.rows[:3])
        with tempfile.TemporaryDirectory() as directory:
            vector = {
                "persist_directory": directory,
                "collection_name": "test_gold_chunks",
                "embedding_model": "dummy/test",
                "add_batch_size": 2,
            }
            store, stats = build_chroma(documents, DummyEmbeddings(), vector, rebuild=True)
            self.assertEqual(stats["gold_documents"], 3)
            self.assertEqual(stats["chroma_documents"], 3)
            self.assertEqual(store._collection.count(), 3)
            reloaded = open_chroma(vector, DummyEmbeddings())
            self.assertEqual(reloaded._collection.count(), 3)
            repeated_store, repeated = build_chroma(documents, DummyEmbeddings(), vector)
            self.assertEqual(repeated["chroma_documents"], 3)
            close_chroma(repeated_store)
            close_chroma(reloaded)
            close_chroma(store)

    def test_retriever_searches_without_relic_filter_and_normalizes(self):
        docs = [
            Document(id="A", page_content="첫 자료", metadata={"chunk_id": "A", "relic_label": "본관 1"}),
            Document(id="B", page_content="둘째 자료", metadata={"chunk_id": "B", "relic_label": "본관 2"}),
        ]
        store = FakeStore(docs)
        results = retrieve(store, "비교", {"top_k": 2})
        self.assertEqual(store.queries, [("비교", 2)])
        self.assertEqual([item["chunk_id"] for item in results], ["A", "B"])
        self.assertEqual(results[1]["score"], 0.9)

    def test_multiturn_rewrite_retrieval_and_history_are_connected(self):
        docs = [Document(
            id="other:profile",
            page_content="다른 유물 자료",
            metadata={"chunk_id": "other:profile", "relic_label": "본관 2"},
        )]
        calls = []

        def fake_generate(messages, _config):
            calls.append(deepcopy(messages))
            if "재작성기" in messages[0]["content"]:
                return "감산사에서 발견된 다른 유물"
            return "도슨트 답변"

        runtime = deepcopy(self.config)
        runtime["data"]["gold_path"] = str(self.gold_path)
        service = DocentChatService(config=runtime, store=FakeStore(docs), generate_fn=fake_generate)
        initial = service.describe_relic(relic_label="본관 1958")
        follow = service.answer_follow_up("그곳에서는 또 어떤 유물이 발견됐어?",
                                          session_id=initial["session_id"])
        self.assertEqual(follow["retrieval_query"], "감산사에서 발견된 다른 유물")
        self.assertEqual(follow["retrieved"][0]["relic_label"], "본관 2")
        session = service.get_session(initial["session_id"])
        self.assertEqual([message["role"] for message in session["history"]],
                         ["assistant", "user", "assistant"])
        self.assertIn("도슨트 답변", calls[1][1]["content"])

    def test_visitor_instruction_is_added_to_answer_system_prompt(self):
        calls = []

        def fake_generate(messages, _config):
            calls.append(deepcopy(messages))
            return "어린이용 설명"

        runtime = deepcopy(self.config)
        runtime["data"]["gold_path"] = str(self.gold_path)
        service = DocentChatService(config=runtime, store=FakeStore([]), generate_fn=fake_generate)
        result = service.describe_relic(relic_label="본관 1958", visitor_type="child")

        self.assertEqual(result["visitor_type"], "child")
        self.assertEqual(calls[0][0]["role"], "system")
        self.assertIn("유물명이 아니라 소장품번호", DOCENT_SYSTEM_PROMPT)
        self.assertIn(VISITOR_INSTRUCTIONS["child"], calls[0][0]["content"])
        self.assertIn(INITIAL_DESCRIPTION_INSTRUCTION, calls[0][0]["content"])
        expected_name = get_relic_records_by_label(self.gold_path, "본관 1958")[0]["text"].split("—", 1)[0].strip()
        self.assertIn(f"현재 유물 '{expected_name}'", calls[0][1]["content"])
        self.assertIn("'본관 1958'은 유물명이 아니라 소장품번호", calls[0][1]["content"])
        with self.assertRaisesRegex(ValueError, "child, general, expert"):
            service.describe_relic(relic_label="본관 1958", visitor_type="test")

    def test_greeting_is_limited_to_first_follow_up(self):
        calls = []

        def fake_generate(messages, _config):
            calls.append(deepcopy(messages))
            return "답변"

        runtime = deepcopy(self.config)
        runtime["data"]["gold_path"] = str(self.gold_path)
        service = DocentChatService(config=runtime, store=FakeStore([]), generate_fn=fake_generate)
        initial = service.describe_relic(relic_label="본관 1958", visitor_type="child")
        service.answer_follow_up("제작 기법은 무엇이야?", session_id=initial["session_id"])
        service.answer_follow_up("재질은 무엇이야?", session_id=initial["session_id"])

        self.assertIn(INITIAL_DESCRIPTION_INSTRUCTION, calls[0][0]["content"])
        self.assertIn(FIRST_FOLLOW_UP_INSTRUCTION, calls[1][0]["content"])
        self.assertNotIn(LATER_FOLLOW_UP_INSTRUCTION, calls[1][0]["content"])
        self.assertIn(LATER_FOLLOW_UP_INSTRUCTION, calls[2][0]["content"])
        self.assertNotIn(FIRST_FOLLOW_UP_INSTRUCTION, calls[2][0]["content"])

    def test_rewrite_falls_back_when_small_model_drops_entities(self):
        runtime = deepcopy(self.config)
        runtime["data"]["gold_path"] = str(self.gold_path)
        service = DocentChatService(
            config=runtime,
            store=FakeStore([]),
            generate_fn=lambda _messages, _config: "이 유물",
        )
        rewritten = service.rewrite_query(
            question="이 유물은 어디에서 발견됐어?",
            current_relic_label="본관 2789",
            current_relic_name="금동 반가 사유상",
            history=[{"role": "assistant", "content": "현재 유물 설명"}],
        )
        self.assertIn("금동 반가 사유상", rewritten)
        self.assertIn("어디에서 발견", rewritten)

    def test_explicit_comparison_retrieves_only_other_target(self):
        runtime = deepcopy(self.config)
        runtime["data"]["gold_path"] = str(self.gold_path)
        service = DocentChatService(config=runtime, store=FakeStore([]),
                                    generate_fn=lambda _messages, _config: "사용되지 않음")
        rewritten = service.rewrite_query(
            question="이 유물과 농경문 청동기를 비교해줘.",
            current_relic_label="본관 2789",
            current_relic_name="금동 반가 사유상",
            history=[],
        )
        self.assertEqual(rewritten, "농경문 청동기")

    def test_recall_and_mrr(self):
        docs = [
            Document(id="X", page_content="x", metadata={"chunk_id": "X", "relic_label": "Lx"}),
            Document(id="A", page_content="a", metadata={"chunk_id": "A", "relic_label": "La"}),
            Document(id="B", page_content="b", metadata={"chunk_id": "B", "relic_label": "Lb"}),
            Document(id="Y", page_content="y", metadata={"chunk_id": "Y", "relic_label": "Ly"}),
            Document(id="Z", page_content="z", metadata={"chunk_id": "Z", "relic_label": "Lz"}),
        ]
        result = evaluate(FakeStore(docs), [{
            "id": "q1", "question": "질문", "category": "test",
            "relevant_chunk_ids": ["A", "B"],
            "relevant_relic_labels": ["La", "Lb"],
        }], {"top_k": 5})
        self.assertEqual(result["recall_at_1"], 0.0)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["mrr"], 0.5)

    def test_evaluation_query_must_not_expose_relic_label(self):
        valid = [{
            "id": "q1",
            "question": "보물 1437호 백자 달항아리의 크기는?",
            "relevant_chunk_ids": ["A"],
            "relevant_relic_labels": ["접수 702"],
        }]
        validate_no_relic_labels_in_queries(valid, {"접수 702"})

        invalid = [{**valid[0], "question": "접수 702의 크기는?"}]
        with self.assertRaisesRegex(ValueError, "relic_label '접수 702'"):
            validate_no_relic_labels_in_queries(invalid, {"접수 702"})


if __name__ == "__main__":
    unittest.main()
