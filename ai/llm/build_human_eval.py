"""Build a blind A/B/C comparison sheet for manual Human Evaluation.

Per PDF 방법론 권장 운영 방식: 모델명을 숨긴 채 답변만 보고 점수/선호도를 기록한다.
Labels are re-shuffled per question so position alone can't reveal the model.

Outputs:
    ai/llm/human_eval_blind.md  - 팀원들이 실제로 보고 채점할 시트 (모델명 없음)
    ai/llm/human_eval_key.json  - 채점 끝난 뒤에만 열어볼 정답표 (label -> model)

Usage:
    python -m ai.llm.build_human_eval
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "comparison_results.json"
BLIND_PATH = Path(__file__).resolve().parent / "human_eval_blind.md"
KEY_PATH = Path(__file__).resolve().parent / "human_eval_key.json"

LABELS = ["A", "B", "C"]


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
                "answers": {},
            }
        groups[key]["answers"][r["model"]] = r["response"]
    return list(groups.values())


def main() -> None:
    results = load_results()
    groups = group_by_question(results)

    key: dict[str, dict[str, str]] = {}
    lines = [
        "# Human Evaluation - 블라인드 비교표",
        "",
        "모델명은 안 보이게 A/B/C로 섞여 있습니다 (문항마다 순서가 다시 섞임).",
        "각 문항에서 **어느 답변이 제일 낫다고 생각하는지**, 그리고 이유 오류·어색함이",
        "있는지 표시해주세요. 다 끝난 뒤에 `human_eval_key.json`을 열어서 실제 모델명을",
        "확인하세요.",
        "",
        "| 문항 | 정확성(1-5) | 자연스러움(1-5) | 선호 (A/B/C/동점) | 비고 |",
        "|---|---|---|---|---|",
        "",
    ]

    for idx, group in enumerate(groups, start=1):
        model_names = list(group["answers"].keys())
        shuffled = LABELS[: len(model_names)]
        random.shuffle(model_names)
        label_to_model = dict(zip(shuffled, model_names))
        question_key = f"Q{idx}_{group['artifact_id']}_{group['question_type']}"
        key[question_key] = label_to_model

        lines.append(f"## {question_key}")
        lines.append("")
        lines.append(f"**유물**: {group['artifact_name']} ({group['artifact_id']})")
        lines.append(f"**질문**: {group['question']}")
        lines.append("")
        for label in shuffled:
            answer = group["answers"][label_to_model[label]]
            lines.append(f"**답변 {label}**")
            lines.append("")
            lines.append(answer)
            lines.append("")
        lines.append("| 정확성(1-5) | 자연스러움(1-5) | 선호(A/B/C/동점) | 비고 |")
        lines.append("|---|---|---|---|")
        lines.append("|  |  |  |  |")
        lines.append("")
        lines.append("---")
        lines.append("")

    BLIND_PATH.write_text("\n".join(lines), encoding="utf-8")
    KEY_PATH.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"questions={len(groups)}")
    print(f"blind_sheet={BLIND_PATH}")
    print(f"answer_key={KEY_PATH} (채점 끝나기 전엔 열어보지 말 것)")


if __name__ == "__main__":
    main()
