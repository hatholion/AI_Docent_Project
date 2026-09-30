"""RAG 경로와 설정값. 초기값은 기획서 §6 표를 따르고 평가(§8)로 조정한다."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "emuseum"
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "data" / "index" / "emuseum")))
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "emuseum_questions.jsonl"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "runs" / "rag"

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "exaone3.5:7.8b")

CHROMA_COLLECTION = "emuseum-chunks"

# 샘플 데이터 범위 안내 문구 (기획서 §2 표본 편향)
TOTAL_COLLECTION_SIZE = 207_494


@dataclass(frozen=True)
class RetrievalSettings:
    k_child: int = 30  # 벡터 / BM25 각각 가져올 청크 수
    n_parent: int = 6  # LLM에 넘길 parent 카드 최대 수
    related_max: int = 2  # 1순위 카드의 연관 소장품 확장 수 (1단계만)
    rrf_k: int = 60  # Reciprocal Rank Fusion 상수
    # 벡터 최고 유사도와 BM25 최고 점수가 모두 이 값 미만이면 "근거 없음".
    # 평가 세트(2026-09-29)에서 정답이 있는 질문의 최저값(벡터 0.479 / BM25 8.85)과
    # 없는 유물 질문의 최고값(벡터 0.487 / BM25 6.08) 사이로 정했다. 질문이 늘면 다시 맞춘다.
    min_vector_similarity: float = 0.50
    min_bm25_score: float = 7.5


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.2
    num_ctx: int = 8192
    max_tokens: int = 700
    timeout_s: float = 180.0
