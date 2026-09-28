"""평가 입력의 소장품번호와 실제 검색 필터의 연결을 검증한다."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ai.llm.evaluation.retrieval_eval import evaluate, load_evaluation


class FakeStore:
    def __init__(self):
        self.filters = []

    def similarity_search(self, query, *, k, filter):
        self.filters.append(filter)
        relic_id = filter["relic_id"]
        return [SimpleNamespace(id=f"{relic_id}:a", metadata={
            "relic_id": relic_id, "chunk_id": f"{relic_id}:a"})]


class EvaluationLabelTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"relic_id": "r1", "relic_label": "본관 2789"},
                     {"relic_id": "r2", "relic_label": "본관 1"}]
        self.store = FakeStore()
        self.config = {"search_type": "similarity", "k": 5}

    def item(self, label="본관 2789", relevant=None):
        return {"query": "재질은 무엇인가요?", "relic_label": label,
                "relevant_chunk_ids": relevant or ["r1:a", "r1:b"]}

    def test_each_label_selects_its_own_relic(self):
        result = evaluate(self.store, [self.item(" 본관 2789 "),
                          self.item("본관 1", ["r2:a"])], self.config,
                          gold_rows=self.rows)
        self.assertEqual(self.store.filters, [{"relic_id": "r1"}, {"relic_id": "r2"}])
        self.assertEqual(result["per_query"][0]["relic_label"], "본관 2789")
        self.assertEqual(result["per_query"][0]["relic_id"], "r1")
        for k in (1, 3, 5):
            self.assertEqual(result["averages"][f"chunk_recall_at_{k}"], 0.75)

    def test_invalid_labels_fail_before_search(self):
        for label in (None, "", " ", 123, "없는 번호"):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "평가 2번.*relic_label"):
                evaluate(self.store, [self.item(), self.item(label)], self.config,
                         gold_rows=self.rows)
        self.assertEqual(self.store.filters, [])

    def test_ambiguous_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "중복"):
            evaluate(self.store, [self.item()], self.config,
                     gold_rows=self.rows + [{"relic_id": "r3", "relic_label": "본관 2789"}])
        self.assertEqual(self.store.filters, [])

    def test_json_schema_requires_label_and_rejects_legacy_id(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            item = self.item()
            path.write_text(json.dumps([item]), encoding="utf-8")
            self.assertEqual(load_evaluation(path), [item])
            del item["relic_label"]
            for legacy in (False, True):
                if legacy:
                    item["relic_id"] = "r1"
                path.write_text(json.dumps([item]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "relic_label"):
                    load_evaluation(path)


if __name__ == "__main__":
    unittest.main()
