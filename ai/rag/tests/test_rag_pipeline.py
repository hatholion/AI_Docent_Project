"""Gold 선택, 선택 유물 검색, 청크 평가, 방문자 프롬프트 회귀 검사."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ai.llm.evaluation.retrieval_eval import evaluate, load_evaluation
from ai.rag.chunkers import chunk_documents
from ai.rag.config import load_config
from ai.rag.document_builder import build_document
from ai.llm.rag_chain import BASE_INSTRUCTION, ask_rag
from ai.rag.retriever import retrieve
from ai.rag.scripts.build_gold_dataset import DATA, TARGET_IDS, load_jsonl, select_artifacts


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FakeStore:
    def __init__(self, docs):
        self.docs = docs
        self.filters = []

    def similarity_search(self, query, k, filter=None):
        self.filters.append(filter)
        return [doc for doc in self.docs if doc.metadata["relic_id"] == filter["relic_id"]][:k]

    def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult, filter=None):
        return self.similarity_search(query, k, filter=filter)


class PipelineTests(unittest.TestCase):
    def test_selected_gold_labels_and_count(self):
        rows = select_artifacts(load_jsonl(DATA / "Silver_0928.jsonl"))
        self.assertEqual({row["relic_label"] for row in rows}, set(TARGET_IDS))
        self.assertEqual(len(rows), len(TARGET_IDS))
        self.assertEqual(len({row["relic_id"] for row in rows}), len(rows))

    def test_chunk_id_is_content_based_and_survives_chroma(self):
        config = {"strategy": "field_based", "chunk_size": 500, "chunk_overlap": 50}
        row = {"relic_id": "r1", "relic_label": "", "name_kr": "이름", "description": "동일한 설명"}
        original = chunk_documents([build_document(row)], config, [row])
        changed = {**row, "name_kr": ""}
        reindexed = chunk_documents([build_document(changed)], config, [changed])
        detail_before = next(doc for doc in original if doc.metadata["chunk_type"] == "detail")
        detail_after = next(doc for doc in reindexed if doc.metadata["chunk_type"] == "detail")
        self.assertNotEqual(detail_before.metadata["chunk_index"], detail_after.metadata["chunk_index"])
        self.assertEqual(detail_before.metadata["chunk_id"], detail_after.metadata["chunk_id"])
        with tempfile.TemporaryDirectory() as directory:
            store = Chroma(collection_name="test_chunks", embedding_function=DummyEmbeddings(),
                           persist_directory=directory)
            other = Document(id="r2:chunk", page_content="동일한 설명",
                             metadata={"relic_id": "r2", "chunk_id": "r2:chunk"})
            docs = [*original, other]
            store.add_documents(docs, ids=[doc.metadata["chunk_id"] for doc in docs])
            for search_type in ("similarity", "mmr"):
                retrieved = retrieve(store, "동일한 설명", {"search_type": search_type, "k": 5,
                                                         "fetch_k": 20, "lambda_mult": 0.5}, relic_id="r1")
                self.assertTrue(retrieved)
                self.assertTrue(all(doc.metadata["relic_id"] == "r1" for doc in retrieved))
                self.assertTrue(all(doc.id == doc.metadata["chunk_id"] for doc in retrieved))

    def test_relic_id_is_required(self):
        with self.assertRaisesRegex(ValueError, "relic_id는 필수"):
            retrieve(FakeStore([]), "질문", {"search_type": "similarity", "k": 5})
        with self.assertRaisesRegex(ValueError, "relic_id는 필수"):
            ask_rag(FakeStore([]), "질문", {"search_type": "similarity", "k": 5},
                    {"enabled": True, "provider": "ollama", "model": "test",
                     "temperature": 0, "visitor_type": "test"})

    def test_chunk_recall_and_dataset_validation(self):
        ids = ["A", "X", "C", "Y", "Z"]
        docs = [Document(id=chunk, page_content=chunk,
                         metadata={"relic_id": "r1", "chunk_id": chunk}) for chunk in ids]
        docs.append(Document(id="OTHER", page_content="other",
                             metadata={"relic_id": "r2", "chunk_id": "OTHER"}))
        store = FakeStore(docs)
        result = evaluate(store, [{"query": "설명", "relic_label": "본관 2789",
                                   "relevant_chunk_ids": ["A", "B", "C"]}],
                          {"search_type": "similarity", "k": 5},
                          gold_rows=[{"relic_id": "r1", "relic_label": "본관 2789"}])
        self.assertEqual(result["per_query"][0]["retrieved_chunk_ids"], ids)
        self.assertEqual(result["averages"], {"chunk_recall_at_1": 1/3,
                                               "chunk_recall_at_3": 2/3,
                                               "chunk_recall_at_5": 2/3})
        self.assertEqual(store.filters, [{"relic_id": "r1"}])
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "bad.json"
            dataset.write_text('[{"query":"질문","relevant_chunk_ids":["A"]}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relic_label"):
                load_evaluation(dataset)

    def test_all_visitor_types_do_not_change_ranking(self):
        docs = [Document(id="a", page_content="자료", metadata={"relic_id": "r1", "chunk_id": "a"})]
        store = FakeStore(docs)
        prompts = {}

        class FakeModel:
            def __init__(self, **kwargs):
                pass

            def invoke(self, messages):
                prompts[current_type] = messages[0].content
                return type("Response", (), {"content": "답변"})()

        with patch("ai.llm.rag_chain.ChatOllama", FakeModel):
            for current_type in ("test", "child", "general", "expert"):
                result = ask_rag(store, "질문", {"search_type": "similarity", "k": 5},
                                 {"enabled": True, "provider": "ollama", "model": "test",
                                  "temperature": 0, "visitor_type": current_type}, relic_id="r1")
                self.assertEqual(result["retrieved_ids"], ["r1"])
                self.assertEqual(result["visitor_type"], current_type)
            with self.assertRaisesRegex(ValueError, "visitor_type"):
                ask_rag(store, "질문", {"search_type": "similarity", "k": 5},
                        {"enabled": True, "provider": "ollama", "model": "test",
                         "temperature": 0, "visitor_type": "invalid"}, relic_id="r1")
        self.assertEqual(store.filters, [{"relic_id": "r1"}] * 4)
        self.assertEqual(prompts["test"], BASE_INSTRUCTION)
        self.assertEqual(len(set(prompts.values())), 4)

    def test_yaml_visitor_type_validation(self):
        self.assertEqual(load_config("ai/llm/configs/base.yaml")["llm"]["visitor_type"], "general")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("llm:\n  visitor_type: invalid\n")
            with self.assertRaisesRegex(ValueError, "visitor_type"):
                load_config(path)

    def test_transformers_receives_rag_messages(self):
        docs = [Document(id="a", page_content="유물 근거", metadata={"relic_id": "r1", "chunk_id": "a"})]
        store = FakeStore(docs)
        config = {"enabled": True, "provider": "transformers", "visitor_type": "general"}
        with patch("ai.llm.rag_chain.generate_answer", return_value="직접 생성") as generate:
            with patch("ai.llm.rag_chain.ChatOllama") as ollama:
                result = ask_rag(store, "재질?", {"search_type": "similarity", "k": 5}, config, relic_id="r1")
        self.assertEqual(result["answer"], "직접 생성")
        self.assertEqual(result["retrieved_ids"], ["r1"])
        messages, passed_config = generate.call_args.args
        self.assertIn("유물 근거", messages[1]["content"])
        self.assertIn("재질?", messages[1]["content"])
        self.assertEqual(passed_config, config)
        ollama.assert_not_called()


if __name__ == "__main__":
    unittest.main()
