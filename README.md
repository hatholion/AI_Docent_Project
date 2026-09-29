# gg_minipjt_2

국립중앙박물관 AI 도슨트 프로젝트.

현재 AI 파이프라인의 실행 방법과 Gold/Chroma/멀티턴 구조는 [ai/README.md](ai/README.md)에 정리되어 있다.

사용자가 유물 사진을 업로드하면 (1) EfficientNet 기반 Vision 모델이 유물 종류를 분류하고,
(2) 분류된 유물 ID로 공식 유물 설명 데이터를 검색하고,
(3) RAG + Local LLM으로 관람객에게 도슨트 설명을 제공하는 서비스를 목표로 한다.

## 프로젝트 구조

```text
gg_minipjt_2/
├─ data/
│  ├─ source_3d/          # 국립중앙박물관 원본 3D 데이터 (OBJ/PLY/STL), Git 추적 제외
│  │  └─ <artifact_id>/
│  │
│  ├─ raw/                # Git 추적 제외
│  │  ├─ synthetic/       # Blender로 생성한 synthetic 학습 이미지
│  │  │  └─ <artifact_id>/
│  │  └─ real/             # 실제 촬영/박물관 이미지 (추후 추가)
│  │     └─ <artifact_id>/
│  │
│  ├─ processed/          # Git 추적 제외, ImageFolder용 train/val/test 분리본
│  │  ├─ train/<artifact_id>/
│  │  ├─ val/<artifact_id>/
│  │  └─ test/<artifact_id>/
│  │
│  └─ metadata.csv        # artifact_id, artifact_name, source, description
│
├─ ai/            # Vision / RAG / LLM 코드
├─ backend/       # API 서버
├─ frontend/      # 클라이언트
├─ docs/          # 공용 스키마, 규칙 문서
├─ .gitignore
├─ Git_규칙.md
└─ README.md
```

유물 클래스가 추가될 때마다 `data/source_3d/`, `data/raw/synthetic/`, `data/raw/real/`,
`data/processed/{train,val,test}/` 아래에 동일한 `<artifact_id>` 폴더 규칙으로 확장한다.

`data/` 하위의 3D 원본, 이미지 데이터셋은 용량 문제로 Git 추적에서 제외된다 (`.gitignore` 참고).
