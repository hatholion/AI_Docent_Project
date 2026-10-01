"""도슨트 라우팅 점검: 기본은 ID 조회, 탐색 질문만 RAG 검색으로 가는지 (LLM 호출 없음)."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.docent import ArtifactNotFound, Docent  # noqa: E402


# (현재 유물, 질문, 기대 route, 근거 카드에 있어야 할 번호, 없어야 할 번호, 기대 필터)
CASES = [
    ("bon002789", "왜 손가락을 뺨에 대고 있어?", "artifact", ["본관 2789"], [], None),
    ("bon002789", "이거 언제 만들어졌어?", "artifact", ["본관 2789"], [], None),
    ("bon002789", "금동 반가사유상은 얼마나 커?", "artifact", ["본관 2789"], [], None),
    ("jub000702", "이 백자 항아리는 어디서 만들었어?", "artifact", ["접수 702"], [], None),
    ("ssu022891", "빗살무늬토기는 어떻게 만들었어?", "artifact", ["신수 22891"], [], None),
    ("bon002789", "신수 1846은 뭐야?", "label", ["신수 1846"], [], None),
    ("bon002789", "신수 999999는 뭐야?", "not_found", [], [], None),
    ("bon002789", "농경문 청동기도 알려줘", "name", ["신수 1794"], [], None),
    ("bon002789", "비슷한 유물 또 있어?", "search", ["본관 2789"], [], None),
    ("bon002789", "같은 시대 다른 유물 보여줘", "search", ["본관 2789"], [], {"period": "삼국"}),
    ("ssu001846", "같은 재질로 만든 다른 유물 추천해줘", "search", ["신수 1846"], [], {"material_l2": "동합금"}),
    ("bon002789", "국립중앙박물관에 불상이 몇 점 있어?", "search", [], [], None),
]


def check(condition: bool, label: str) -> bool:
    print(f"[{'OK' if condition else 'FAIL'}] {label}")
    return condition


def main() -> None:
    docent = Docent()
    ok = True

    same = {docent.resolve(ref) for ref in ("bon002789", "본관 2789", "본관2789", "nmm-bon-002789-00")}
    ok &= check(same == {"nmm-bon-002789-00"}, "resolve: artifact_id / 소장품 번호 / relic_id")
    try:
        docent.resolve("no_such_artifact")
        ok &= check(False, "resolve: 없는 유물은 ArtifactNotFound")
    except ArtifactNotFound:
        ok &= check(True, "resolve: 없는 유물은 ArtifactNotFound")

    for ref, question, route, must, must_not, filters in CASES:
        result = docent.route_chat(docent.resolve(ref), question)
        labels = [p.label for p in result.retrieval.parents]
        passed = result.route == route and all(l in labels for l in must) and not any(l in labels for l in must_not)
        if filters:
            passed &= all(result.retrieval.filters.get(k) == v for k, v in filters.items())
        detail = f"{result.route} ({result.reason}) filters={result.retrieval.filters} cards={labels[:4]}"
        ok &= check(passed, f"{ref} '{question}' → {detail}")
        if route == "search" and "통계" not in result.reason:
            # 탐색 결과에는 현재 유물 외의 소장품이 있어야 한다
            ok &= check(len(labels) > 1 and labels.count(must[0]) == 1, "  탐색 결과에 다른 소장품 포함, 현재 유물은 1번만")

    print("docent routing check passed" if ok else "docent routing check FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
