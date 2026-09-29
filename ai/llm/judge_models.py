"""LLM-as-a-Judge scoring for the 3-model comparison (ai/llm/compare_models.py).

For each Golden Set question, the 3 candidate answers are shown to a judge model
in randomized, anonymized order (A/B/C) so the judge can't tell which model wrote
which answer. Every one of the 3 models also takes a turn as judge (round robin),
so no single judge's self-preference bias dominates the aggregate score.

Criteria (1-5 each, matches the reference eval methodology):
    correctness   - 주어진 [유물 정보]와 실제로 맞는 내용인가
    relevance     - 질문에 실제로 답했는가
    groundedness  - 정보에 없는 내용을 지어내지 않았는가 (hallucination 방지)
    fluency       - 문장이 자연스러운가

Usage:
    python -m ai.llm.judge_models
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ai.llm.ollama_client import generate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = Path(__file__).resolve().parent / "comparison_results.json"
JUDGE_OUTPUT_PATH = Path(__file__).resolve().parent / "judge_results.json"
JUDGE_SUMMARY_PATH = Path(__file__).resolve().parent / "judge_summary.md"

JUDGES = ["exaone3.5:7.8b", "llama3.1:latest", "qwen2.5-7b-quant-manual"]
CRITERIA = ["correctness", "relevance", "groundedness", "fluency"]
LABELS = ["A", "B", "C"]

JUDGE_PROMPT_TEMPLATE = """당신은 박물관 도슨트 챗봇 답변을 채점하는 엄격한 평가자입니다.
아래 [유물 정보]에 있는 사실만 근거로, 세 답변(A/B/C)을 각 기준마다 1~5점으로 채점하세요.
어느 답변이 어떤 AI가 만든 것인지는 알려주지 않습니다. 순서로 우열을 추측하지 마세요.

[유물 정보]
{context_text}

[관람객 질문]
{question}

[답변 A]
{answer_a}

[답변 B]
{answer_b}

[답변 C]
{answer_c}

[채점 기준]
- correctness: 유물 정보와 실제로 일치하는 내용인가 (1=틀림, 5=정확함)
- relevance: 질문에 실제로 답했는가 (1=동문서답, 5=정확히 답함)
- groundedness: 유물 정보에 없는 내용을 지어내지 않았는가 (1=지어냄, 5=근거 있는 내용만)
- fluency: 문장이 자연스러운가 (1=어색함, 5=자연스러움)

아래 JSON 형식으로만 답하세요. 다른 텍스트는 쓰지 마세요.
{{"A": {{"correctness": 0, "relevance": 0, "groundedness": 0, "fluency": 0}},
 "B": {{"correctness": 0, "relevance": 0, "groundedness": 0, "fluency": 0}},
 "C": {{"correctness": 0, "relevance": 0, "groundedness": 0, "fluency": 0}}}}"""


def load_results() -> list[dict]:
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def group_by_question(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r["artifact_id"], r["question_type"])
        if key not in groups:
            groups[key] = {
                "artifact_id": r["artifact_id"],
                "artifact_name": r["artifact_name"],
                "question_type": r["question_type"],
                "question": r["question"],
                "context_text": None,  # filled below
                "answers": {},  # model -> response text
            }
        groups[key]["answers"][r["model"]] = r["response"]
    return list(groups.values())


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_one(judge_model: str, group: dict, context_by_id: dict[str, str]) -> dict:
    model_names = list(group["answers"].keys())
    shuffled_labels = LABELS[: len(model_names)]
    random.shuffle(model_names)
    label_to_model = dict(zip(shuffled_labels, model_names))

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        context_text=context_by_id[group["artifact_id"]],
        question=group["question"],
        answer_a=group["answers"][label_to_model["A"]],
        answer_b=group["answers"][label_to_model["B"]],
        answer_c=group["answers"][label_to_model["C"]],
    )
    response = generate(prompt, judge_model, options={"temperature": 0})
    raw = response.get("response", "")
    try:
        scores_by_label = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "judge": judge_model,
            "artifact_id": group["artifact_id"],
            "question_type": group["question_type"],
            "error": str(error),
            "raw": raw,
        }

    scores_by_model = {
        label_to_model[label]: scores_by_label.get(label, {}) for label in shuffled_labels
    }
    return {
        "judge": judge_model,
        "artifact_id": group["artifact_id"],
        "question_type": group["question_type"],
        "label_to_model": label_to_model,
        "scores_by_model": scores_by_model,
    }


def main() -> None:
    results = load_results()
    groups = group_by_question(results)

    from ai.rag.gold_store import get_context_text

    context_by_id = {g["artifact_id"]: get_context_text(g["artifact_id"]) for g in groups}

    judgments = []
    for judge_model in JUDGES:
        for group in groups:
            verdict = judge_one(judge_model, group, context_by_id)
            judgments.append(verdict)
            status = "ERROR" if "error" in verdict else "ok"
            print(f"[judge={judge_model}] {group['artifact_id']}/{group['question_type']}: {status}")

    JUDGE_OUTPUT_PATH.write_text(
        json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Aggregate: model -> criterion -> list of scores
    totals: dict[str, dict[str, list[float]]] = {}
    win_counts: dict[str, int] = {}
    tie_count = 0
    valid_judgments = [j for j in judgments if "error" not in j]

    for j in valid_judgments:
        per_model_avg = {}
        for model, scores in j["scores_by_model"].items():
            totals.setdefault(model, {c: [] for c in CRITERIA})
            for criterion in CRITERIA:
                value = scores.get(criterion)
                if isinstance(value, (int, float)):
                    totals[model][criterion].append(value)
            valid_scores = [scores.get(c) for c in CRITERIA if isinstance(scores.get(c), (int, float))]
            if valid_scores:
                per_model_avg[model] = sum(valid_scores) / len(valid_scores)
        if per_model_avg:
            best_score = max(per_model_avg.values())
            winners = [m for m, s in per_model_avg.items() if s == best_score]
            if len(winners) == 1:
                win_counts[winners[0]] = win_counts.get(winners[0], 0) + 1
            else:
                tie_count += 1

    lines = [
        "# LLM Judge 결과 (3개 모델이 돌아가며 심판, temperature=0)",
        "",
        f"유효 채점: {len(valid_judgments)} / {len(judgments)} (JSON 파싱 실패 제외)",
        "",
        "| 모델 | correctness | relevance | groundedness | fluency | 평균 | 승수(무승부 제외) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in JUDGES:
        crit_avgs = []
        row = []
        for criterion in CRITERIA:
            scores = totals.get(model, {}).get(criterion, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            crit_avgs.append(avg)
            row.append(f"{avg:.2f}")
        overall = sum(crit_avgs) / len(crit_avgs) if crit_avgs else 0.0
        wins = win_counts.get(model, 0)
        lines.append(f"| {model} | " + " | ".join(row) + f" | {overall:.2f} | {wins} |")
    lines.append("")
    lines.append(f"무승부(동점): {tie_count}건")

    summary = "\n".join(lines)
    JUDGE_SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nraw_output={JUDGE_OUTPUT_PATH}")
    print(f"summary_output={JUDGE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
