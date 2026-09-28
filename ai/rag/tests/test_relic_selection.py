"""소장품번호 입력이 실행·검색·LLM 호출까지 올바른 내부 ID로 전달되는지 검사한다."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai.llm.evaluation import run_experiment
from ai.rag.config import load_config
from ai.rag.gold_loader import load_gold, resolve_relic_id
from ai.rag.scripts import test_retriever
from ai.rag.tests.test_rag_pipeline import FakeStore


class RelicSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("ai/llm/configs/experiments/exp01_recursive_500.yaml")
        self.rows = [
            {"relic_id": "r1", "relic_label": "본관 2789", "name_kr": "첫 유물"},
            {"relic_id": "r2", "relic_label": "덕수 798", "name_kr": "둘째 유물"},
        ]

    def test_label_resolution_and_invalid_selection(self):
        self.assertEqual(resolve_relic_id(self.rows, "  본관 2789  "), "r1")
        for label in (None, "", "  ", 2789, "없는 번호", "본관2789"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                resolve_relic_id(self.rows, label)
        with self.assertRaisesRegex(ValueError, "중복"):
            resolve_relic_id([self.rows[0], {**self.rows[0], "relic_id": "r3"}], "본관 2789")

    def test_config_validation_and_current_gold(self):
        self.assertNotIn("relic_id", self.config["rag"])
        from ai.rag.config import ROOT
        rows = load_gold(ROOT / self.config["data"]["gold_file"])
        self.assertEqual(resolve_relic_id(rows, self.config["rag"]["relic_label"]), "nmm-bon-002789-00")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for fragment in ('relic_label: 123', 'relic_label: "  "', 'relic_id: r1'):
                with self.subTest(fragment=fragment):
                    path.write_text(f"rag:\n  {fragment}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "relic_label"):
                        load_config(path)

    def test_run_uses_yaml_label_and_explicit_override(self):
        self.config["rag"]["smoke_query"] = "이 유물의 재질은 무엇인가요?"
        for label, expected in ((None, "r1"), ("덕수 798", "r2")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                stack.enter_context(patch.object(run_experiment, "ROOT", root))
                stack.enter_context(patch.object(run_experiment, "load_config", return_value=deepcopy(self.config)))
                stack.enter_context(patch.object(run_experiment, "load_gold", return_value=self.rows))
                stack.enter_context(patch.object(run_experiment, "get_embedding"))
                store = FakeStore([])
                stack.enter_context(patch.object(run_experiment, "create_chroma", return_value=store))
                result = run_experiment.run("unused", None, False, relic_label=label)
                self.assertEqual(result["relic_id"], expected)
                self.assertEqual(result["smoke_query"], self.config["rag"]["smoke_query"])
                self.assertEqual(store.filters, [{"relic_id": expected}])
                self.assertEqual(result["relic_label"], label or "본관 2789")
                output = root / "ai/llm/evaluation/results" / f"{self.config['experiment']['name']}.json"
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
                with self.assertRaisesRegex(ValueError, "Gold 범위"):
                    run_experiment.run("unused", 1, False, relic_label="덕수 798")

    def test_search_cli_passes_resolved_id_to_search_and_llm(self):
        for ask in (False, True):
            for override, expected in (([], "r1"), (["--relic-label", "덕수 798"], "r2")):
                with self.subTest(ask=ask, override=override), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    root = Path(directory)
                    (root / self.config["vectorstore"]["base_dir"] / self.config["experiment"]["name"] / "chroma").mkdir(parents=True)
                    stack.enter_context(patch.object(test_retriever, "ROOT", root))
                    stack.enter_context(patch.object(test_retriever, "load_config", return_value=deepcopy(self.config)))
                    stack.enter_context(patch.object(test_retriever, "load_gold", return_value=self.rows))
                    stack.enter_context(patch.object(test_retriever, "get_embedding"))
                    stack.enter_context(patch.object(test_retriever, "Chroma"))
                    search = stack.enter_context(patch.object(test_retriever, "retrieve", return_value=[]))
                    llm = stack.enter_context(patch.object(test_retriever, "ask_rag", return_value={}))
                    stack.enter_context(patch("builtins.print"))
                    argv = ["test_retriever", "--config", "unused", "--query", "질문", *override]
                    if ask:
                        argv.append("--ask")
                    stack.enter_context(patch("sys.argv", argv))
                    test_retriever.main()
                    called = llm if ask else search
                    self.assertEqual(called.call_args.kwargs["relic_id"], expected)


if __name__ == "__main__":
    unittest.main()
