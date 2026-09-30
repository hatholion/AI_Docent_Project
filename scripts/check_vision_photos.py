"""실제 사진 폴더로 Vision 체크포인트를 점검한다.

data/raw/real/<artifact_id>/*.jpg  → 정답이 artifact_id인 사진
data/raw/real/other/*.jpg          → (선택) 7종이 아닌 사진. 기준값 미만으로 거부돼야 정답

python scripts/check_vision_photos.py
python scripts/check_vision_photos.py --threshold 0.7 --photos 다른폴더

결과: 유물별 정답률, 일치도 분포, 틀린 사진 목록, 기준값별 통과율 + runs/vision/photo_check/<시각>.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.vision.inference import ArtifactPredictor  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OTHER = "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, default=PROJECT_ROOT / "data" / "raw" / "real")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "vision" / "fine_tune_gpu_v2" / "best_model.pth")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = ArtifactPredictor.from_paths(args.checkpoint, args.metadata, args.device)
    classes = list(predictor.class_to_idx)
    unknown = sorted(p.name for p in args.photos.iterdir() if p.is_dir() and p.name not in classes and p.name != OTHER)
    if unknown:
        print(f"[주의] 체크포인트에 없는 폴더는 건너뜀: {unknown}")

    rows: list[dict] = []
    for folder in sorted(p for p in args.photos.iterdir() if p.is_dir() and (p.name in classes or p.name == OTHER)):
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                top = predictor.predict_path(image, top_k=2)
            except Exception as error:  # 깨진 파일 등
                print(f"[읽기 실패] {image}: {error}")
                continue
            rows.append({
                "expected": folder.name,
                "file": image.name,
                "predicted": top[0].artifact_id,
                "confidence": round(top[0].confidence, 4),
                "second": top[1].artifact_id,
                "second_confidence": round(top[1].confidence, 4),
            })

    names = {c: predictor.artifact_names[c] for c in classes}
    print(f"\n체크포인트 {args.checkpoint.name} (epoch {predictor.checkpoint_epoch}), 사진 {len(rows)}장, 기준값 {args.threshold:.0%}\n")
    print(f"{'artifact_id':12s} {'이름':10s} {'장수':>4s} {'정답':>6s} {'통과':>6s} {'일치도 최저/중간':>16s}")
    for cls in classes:
        group = [r for r in rows if r["expected"] == cls]
        if not group:
            print(f"{cls:12s} {names[cls]:10s} {0:4d}   (사진 없음)")
            continue
        correct = [r for r in group if r["predicted"] == cls]
        passed = [r for r in correct if r["confidence"] >= args.threshold]
        confs = [r["confidence"] for r in correct] or [0.0]
        print(f"{cls:12s} {names[cls]:10s} {len(group):4d} {len(correct):3d}/{len(group):<2d} {len(passed):3d}/{len(group):<2d}"
              f"   {min(confs):6.1%} / {statistics.median(confs):6.1%}")

    in_class = [r for r in rows if r["expected"] != OTHER]
    others = [r for r in rows if r["expected"] == OTHER]
    wrong = [r for r in in_class if r["predicted"] != r["expected"]]
    low = [r for r in in_class if r["predicted"] == r["expected"] and r["confidence"] < args.threshold]
    wrong_accepted = [r for r in wrong if r["confidence"] >= args.threshold]
    if in_class:
        print(f"\n정답 {len(in_class) - len(wrong)}/{len(in_class)}  |  기준값 통과 정답 {len(in_class) - len(wrong) - len(low)}/{len(in_class)}"
              f"  |  틀렸는데 통과(잘못된 설명이 나감) {len(wrong_accepted)}")
    if others:
        false_accept = [r for r in others if r["confidence"] >= args.threshold]
        print(f"7종 밖 사진 {len(others)}장 중 잘못 통과 {len(false_accept)}장")

    if wrong or low:
        print("\n[틀리거나 기준 미달]")
        for r in sorted(wrong + low, key=lambda r: r["confidence"], reverse=True):
            tag = "오인식" if r["predicted"] != r["expected"] else "일치도 낮음"
            print(f"  {tag:6s} {r['expected']}/{r['file']} → {r['predicted']} {r['confidence']:.1%} (2위 {r['second']} {r['second_confidence']:.1%})")
    for r in others:
        if r["confidence"] >= args.threshold:
            print(f"  7종밖통과 other/{r['file']} → {r['predicted']} {r['confidence']:.1%}")

    print("\n[기준값별]  정답 통과율 / 오인식 통과 / 7종 밖 통과")
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        ok = sum(1 for r in in_class if r["predicted"] == r["expected"] and r["confidence"] >= threshold)
        bad = sum(1 for r in wrong if r["confidence"] >= threshold)
        fa = sum(1 for r in others if r["confidence"] >= threshold)
        print(f"  {threshold:.0%}: {ok}/{len(in_class)}  /  {bad}  /  {fa}/{len(others)}")

    out_dir = PROJECT_ROOT / "runs" / "vision" / "photo_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["expected"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
