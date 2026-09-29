"""최종 Gold → Chroma → 멀티턴 도슨트 파이프라인 설정."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml

from ai.llm.transformers_backend import validate_transformers_config

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "ai/llm/configs/base.yaml"
VISITOR_TYPES = {"child", "general", "expert"}


def deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _mapping(config: dict, name: str) -> dict:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} 설정 객체가 필요합니다")
    return value


def _non_empty_string(section: dict, key: str, dotted: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dotted}에 비어 있지 않은 문자열이 필요합니다")
    section[key] = value.strip()
    return section[key]


def _positive_int(section: dict, key: str, dotted: str, *, allow_zero: bool = False) -> int:
    value = section.get(key)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "0 이상" if allow_zero else "양수"
        raise ValueError(f"{dotted}은 {comparator}의 정수여야 합니다")
    return value


def _finite_number(section: dict, key: str, dotted: str, *, minimum: float = 0.0) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{dotted}은 {minimum} 이상의 유한한 숫자여야 합니다")
    return float(value)


def validate_config(config: dict, *, require_gold: bool = True) -> dict:
    data = _mapping(config, "data")
    gold_path = project_path(_non_empty_string(data, "gold_path", "data.gold_path"))
    if require_gold and not gold_path.is_file():
        raise FileNotFoundError(f"Gold dataset을 찾을 수 없습니다: {gold_path}")

    runtime = _mapping(config, "runtime")
    initial = runtime.get("initial_relic_label")
    if initial is not None and (not isinstance(initial, str) or not initial.strip()):
        raise ValueError("runtime.initial_relic_label은 null 또는 비어 있지 않은 문자열이어야 합니다")
    if isinstance(initial, str):
        runtime["initial_relic_label"] = initial.strip()

    embedding = _mapping(config, "embedding")
    _non_empty_string(embedding, "model", "embedding.model")
    _non_empty_string(embedding, "device", "embedding.device")
    _positive_int(embedding, "batch_size", "embedding.batch_size")
    if not isinstance(embedding.get("normalize_embeddings"), bool):
        raise ValueError("embedding.normalize_embeddings는 true 또는 false여야 합니다")
    for key in ("query_prefix", "document_prefix"):
        if not isinstance(embedding.get(key, ""), str):
            raise ValueError(f"embedding.{key}는 문자열이어야 합니다")

    vector = _mapping(config, "vector_db")
    if vector.get("provider") != "chroma":
        raise ValueError("vector_db.provider는 chroma여야 합니다")
    persist = project_path(_non_empty_string(vector, "persist_directory", "vector_db.persist_directory"))
    if persist == ROOT or ROOT not in persist.parents:
        raise ValueError("vector_db.persist_directory는 프로젝트 내부의 전용 하위 폴더여야 합니다")
    _non_empty_string(vector, "collection_name", "vector_db.collection_name")
    _positive_int(vector, "add_batch_size", "vector_db.add_batch_size")

    retrieval = _mapping(config, "retrieval")
    _positive_int(retrieval, "top_k", "retrieval.top_k")
    chat = _mapping(config, "chat")
    _positive_int(chat, "history_turns", "chat.history_turns", allow_zero=True)
    rewrite = _mapping(config, "query_rewrite")
    if not isinstance(rewrite.get("enabled"), bool):
        raise ValueError("query_rewrite.enabled는 true 또는 false여야 합니다")

    llm = _mapping(config, "llm")
    if not isinstance(llm.get("enabled"), bool):
        raise ValueError("llm.enabled는 true 또는 false여야 합니다")
    if llm.get("provider") not in {"ollama", "transformers"}:
        raise ValueError("llm.provider는 ollama 또는 transformers여야 합니다")
    _non_empty_string(llm, "model", "llm.model")
    visitor_types = llm.get("visitor_types")
    if (not isinstance(visitor_types, list)
            or any(not isinstance(value, str) for value in visitor_types)
            or len(visitor_types) != len(set(visitor_types))
            or set(visitor_types) != VISITOR_TYPES):
        raise ValueError("llm.visitor_types는 child, general, expert만 한 번씩 포함해야 합니다")
    if llm.get("visitor_type") not in visitor_types:
        raise ValueError("llm.visitor_type은 llm.visitor_types 중 하나여야 합니다")
    _finite_number(llm, "temperature", "llm.temperature")
    if llm["provider"] == "ollama":
        top_p = _finite_number(llm, "top_p", "llm.top_p")
        if top_p > 1:
            raise ValueError("llm.top_p는 0 이상 1 이하여야 합니다")
        _positive_int(llm, "num_predict", "llm.num_predict")
        _positive_int(llm, "num_ctx", "llm.num_ctx")
        if not isinstance(llm.get("reasoning"), bool):
            raise ValueError("llm.reasoning은 true 또는 false여야 합니다")
    else:
        validate_transformers_config(llm)
    return config


def load_config(config_file: str | Path | None = None, *, require_gold: bool = True) -> dict:
    """기본 YAML 위에 선택 YAML을 병합하고 즉시 검증한다."""
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"기본 YAML이 객체가 아닙니다: {BASE_CONFIG}")
    if config_file is None or project_path(config_file) == BASE_CONFIG.resolve():
        config = base
    else:
        path = project_path(config_file)
        if not path.is_file():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
        overrides: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(overrides, dict):
            raise ValueError("YAML 설정은 객체여야 합니다")
        config = deep_merge(base, overrides)
    return validate_config(config, require_gold=require_gold)
