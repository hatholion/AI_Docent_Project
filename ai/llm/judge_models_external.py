"""External, neutral LLM-as-a-Judge using an OpenAI model (dev-time tool only).

Fixes the self-preference bias found in ai/llm/judge_models.py (where Llama 3.1
scored its own fabricated 지정번호 5/5 groundedness): this judge is not one of
the 3 candidate models, so it has no reason to favor any of them.

Requires OPENAI_API_KEY in .env (see .env.example) or the environment.

Usage:
    python -m ai.llm.judge_models_external
    python -m ai.llm.judge_models_external --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.llm.judge_models import (
    CRITERIA,
    JUDGE_PROMPT_TEMPLATE,
    LABELS,
    extract_json,
    group_by_question,
    load_results,
)
from ai.llm.openai_client import chat

RESULTS_PATH = Path(__file__).resolve().parent / "comparison_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "judge_external_results.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "judge_external_summary.md"

DEFAULT_MODEL = "gpt-5-mini"


def judge_one_external(model: str, group: dict, context_by_id: dict[str, str]) -> dict:
    import random

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
    raw = chat(prompt, model, temperature=None)
    try:
        scores_by_label = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "judge": model,
            "artifact_id": group["artifact_id"],
            "question_type": group["question_type"],
            "error": str(error),
            "raw": raw,
        }

    scores_by_model = {
        label_to_model[label]: scores_by_label.get(label, {}) for label in shuffled_labels
    }
    return {
        "judge": model,
        "artifact_id": group["artifact_id"],
        "question_type": group["question_type"],
        "label_to_model": label_to_model,
        "scores_by_model": scores_by_model,
    }


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    from ai.rag.gold_store import get_context_text

    results = load_results()
    groups = group_by_question(results)
    context_by_id = {g["artifact_id"]: get_context_text(g["artifact_id"]) for g in groups}
    candidate_models = sorted({r["model"] for r in results})

    judgments = []
    for group in groups:
        verdict = judge_one_external(args.model, group, context_by_id)
        judgments.append(verdict)
        status = "ERROR" if "error" in verdict else "ok"
        print(f"[judge={args.model}] {group['artifact_id']}/{group['question_type']}: {status}")

    OUTPUT_PATH.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")

    totals: dict[str, dict[str, list[float]]] = {m: {c: [] for c in CRITERIA} for m in candidate_models}
    win_counts: dict[str, int] = {}
    tie_count = 0
    valid = [j for j in judgments if "error" not in j]

    for j in valid:
        per_model_avg = {}
        for model, scores in j["scores_by_model"].items():
            for criterion in CRITERIA:
                value = scores.get(criterion)
                if isinstance(value, (int, float)):
                    totals[model][criterion].append(value)
            valid_scores = [scores.get(c) for c in CRITERIA if isinstance(scores.get(c), (int, float))]
            if valid_scores:
                per_model_avg[model] = sum(valid_scores) / len(valid_scores)
        if per_model_avg:
            best = max(per_model_avg.values())
            winners = [m for m, s in per_model_avg.items() if s == best]
            if len(winners) == 1:
                win_counts[winners[0]] = win_counts.get(winners[0], 0) + 1
            else:
                tie_count += 1

    lines = [
        f"# 외부 중립 심판({args.model}) 채점 결과",
        "",
        f"유효 채점: {len(valid)} / {len(judgments)}",
        "",
        "| 모델 | correctness | relevance | groundedness | fluency | 평균 | 승수 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in candidate_models:
        row = []
        crit_avgs = []
        for criterion in CRITERIA:
            scores = totals[model][criterion]
            avg = sum(scores) / len(scores) if scores else 0.0
            crit_avgs.append(avg)
            row.append(f"{avg:.2f}")
        overall = sum(crit_avgs) / len(crit_avgs) if crit_avgs else 0.0
        lines.append(f"| {model} | " + " | ".join(row) + f" | {overall:.2f} | {win_counts.get(model, 0)} |")
    lines.append("")
    lines.append(f"무승부(동점): {tie_count}건")

    summary = "\n".join(lines)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nraw_output={OUTPUT_PATH}")
    print(f"summary_output={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
