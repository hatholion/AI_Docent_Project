"""사진 한 장으로 Vision 인식 → 유물 설명(ID 조회) → 추가 질문(RAG 보조)까지 이어서 돌려본다.

ai/vision 과 ai/rag 는 서로 import 하지 않고, 둘을 잇는 일은 이 스크립트(나중에는 backend/services)가 한다.

python scripts/demo_vision_rag.py --image 사진.jpg
python scripts/demo_vision_rag.py --image 사진.jpg -q "언제 만들어졌어?" -q "비슷한 유물 또 있어?"
python scripts/demo_vision_rag.py --fake-artifact jub000702 --no-llm   # 체크포인트 없이 연결만 확인
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.docent import ArtifactNotFound, Docent  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "vision" / "fine_tune_gpu_v2" / "best_model.pth"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="인식할 유물 사진")
    source.add_argument("--fake-artifact", help="Vision 없이 이 artifact_id로 인식됐다고 가정 (예: jub000702)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.60, help="이 일치도 미만이면 인식 실패 (API와 같은 기본값)")
    parser.add_argument("-q", "--question", action="append", default=[], help="설명 뒤에 이어서 물을 질문 (여러 번 가능)")
    parser.add_argument("--visitor-type", choices=("child", "general", "expert"), default="general")
    parser.add_argument("--no-llm", action="store_true", help="답변 생성 없이 인식 결과와 라우팅·근거 카드만 출력")
    return parser.parse_args()


def recognize(args: argparse.Namespace) -> str | None:
    """사진 → artifact_id. 일치도가 기준 미만이면 None."""
    if args.fake_artifact:
        print(f"[Vision] (가짜) artifact_id={args.fake_artifact}")
        return args.fake_artifact

    if not args.image.is_file():
        raise SystemExit(f"사진 파일이 없습니다: {args.image}")
    # torch는 무거우므로 실제 사진을 넣을 때만 불러온다
    from ai.vision.inference import ArtifactPredictor

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"체크포인트가 없습니다: {args.checkpoint}\n"
            "Vision 담당에게 best_model.pth를 받아 이 경로에 두거나 --checkpoint로 지정하세요."
        )
    started = time.perf_counter()
    predictor = ArtifactPredictor.from_paths(args.checkpoint, args.metadata, args.device)
    predictions = predictor.predict_path(args.image, top_k=min(3, len(predictor.class_to_idx)))
    print(f"[Vision] {args.image.name} ({time.perf_counter() - started:.1f}s, device={predictor.device})")
    for rank, prediction in enumerate(predictions, 1):
        print(f"  {rank}. {prediction.artifact_id} {prediction.artifact_name} {prediction.confidence:.1%}")

    best = predictions[0]
    if best.confidence < args.threshold:
        print(f"→ 일치도 {best.confidence:.1%} < 기준 {args.threshold:.0%}: 유물을 인식하지 못했어요. 다시 촬영해주세요.")
        return None
    return best.artifact_id


def print_answer(answer) -> None:
    print(f"\n{answer.answer}\n")
    for source in answer.sources:
        print(f"  출처: {source}")
    print(f"  (route={answer.route}, LLM {answer.llm_seconds}s)")


def main() -> None:
    args = parse_args()
    artifact_id = recognize(args)
    if artifact_id is None:
        raise SystemExit(1)

    docent = Docent(metadata_csv=args.metadata)
    try:
        relic_id = docent.resolve(artifact_id)
    except ArtifactNotFound as error:
        raise SystemExit(f"[연결 실패] {error} — metadata.csv의 accession_no를 확인하세요.")
    card = docent.gold.relic_cards[relic_id]
    print(f"[연결] {artifact_id} → {card['relic_label']} {card['title']} ({relic_id})")

    print("\n[설명] route=artifact (ID 조회)")
    if args.no_llm:
        print("  근거 카드:", [p.label or p.parent_id for p in docent.artifact_parents(relic_id)])
    else:
        print_answer(docent.describe(artifact_id, visitor_type=args.visitor_type))

    history: list[dict[str, str]] = []
    for question in args.question:
        print(f"\n[질문] {question}")
        if args.no_llm:
            route = docent.route_chat(relic_id, question)
            print(f"  route={route.route} ({route.reason}) 근거 카드:",
                  [p.label or p.parent_id for p in route.retrieval.parents])
            continue
        answer, _ = docent.chat(artifact_id, question, visitor_type=args.visitor_type, history=history)
        print_answer(answer)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.answer}]


if __name__ == "__main__":
    main()
