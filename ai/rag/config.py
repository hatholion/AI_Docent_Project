"""실험 설정 로딩과 검증."""

from copy import deepcopy
from pathlib import Path
import yaml
from ai.rag.gold_loader import require_relic_label
from ai.llm.transformers_backend import validate_transformers_config

ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(experiment_file: str | Path) -> dict:
    base = yaml.safe_load((ROOT / "ai/llm/configs/base.yaml").read_text(encoding="utf-8"))
    experiment_path = Path(experiment_file)
    if not experiment_path.is_absolute():
        experiment_path = ROOT / experiment_path
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    config = deep_merge(base, experiment)
    if config["llm"].get("provider") not in ("ollama", "transformers"):
        raise ValueError("llm.provider는 ollama 또는 transformers여야 합니다")
    if config["llm"]["provider"] == "transformers":
        validate_transformers_config(config["llm"])
    name = config["experiment"]["name"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("experiment.name은 단일 안전한 디렉터리 이름이어야 합니다")
    chunk = config["chunking"]
    if chunk["strategy"] not in {"none", "recursive", "sentence_based", "field_based"}:
        raise ValueError("지원하지 않는 chunking.strategy")
    if chunk["chunk_size"] <= 0 or not 0 <= chunk["chunk_overlap"] < chunk["chunk_size"]:
        raise ValueError("chunk_size/overlap 설정 오류")
    if config["vectorstore"]["type"] != "chroma":
        raise ValueError("지원하지 않는 vectorstore.type")
    if set(config["evaluation"]["metrics"]) - {"chunk_recall_at_1", "chunk_recall_at_3", "chunk_recall_at_5"}:
        raise ValueError("지원하지 않는 evaluation.metrics")
    if config["retriever"]["search_type"] not in {"similarity", "mmr"}:
        raise ValueError("지원하지 않는 retriever.search_type")
    if config["llm"].get("visitor_type") not in {"test", "child", "general", "expert"}:
        raise ValueError("llm.visitor_type은 test, child, general, expert 중 하나여야 합니다")
    rag = config.get("rag", {})
    if not isinstance(rag, dict):
        raise ValueError("rag는 설정 객체여야 합니다")
    if "relic_id" in rag:
        raise ValueError("rag.relic_id 대신 rag.relic_label에 소장품번호를 지정하세요")
    if rag.get("relic_label") is not None:
        rag["relic_label"] = require_relic_label(rag["relic_label"])
    query = rag.get("smoke_query")
    if query is not None:
        if not isinstance(query, str):
            raise ValueError("rag.smoke_query는 질문 문자열 또는 null이어야 합니다")
        rag["smoke_query"] = query.strip() or None
    return config
