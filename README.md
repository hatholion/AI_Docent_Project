# gg_minipjt_2

국립중앙박물관 AI 도슨트 프로젝트.

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
│  ├─ metadata.csv        # artifact_id, accession_no, artifact_name, source, description
│  │
│  └─ rag/                # RAG 코퍼스 (e뮤지엄 크롤링), Git 추적 제외, 구글 드라이브로 관리
│     ├─ raw/emuseum/      # API 원본 응답 (RAW_*.jsonl)
│     ├─ silver/emuseum/   # 소장품 1건 = 1행으로 정제 (Silver_*.jsonl)
│     └─ gold/emuseum/     # RAG 검색용 최종 문서 (아직 비어있음)
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

`data/` 하위의 3D 원본, 이미지 데이터셋, RAG 코퍼스는 용량 문제로 Git 추적에서 제외된다
(`.gitignore` 참고). `data/rag/`는 Vision 학습 데이터와 성격이 달라(전체 소장품 텍스트
코퍼스) 별도로 관리한다.

## 데이터셋 다운로드

`data/source_3d/`, `data/raw/`(Vision 이미지)는 구글 드라이브에서 받아 동일한 경로에 배치한다.

- 다운로드: https://drive.google.com/drive/folders/18_m7ZA4wv9EZ8zzGV0iuktt1gSmp_wYn
- 배치 위치: `source_3d/` → `data/source_3d/`, `raw/` → `data/raw/`

`data/rag/`(e뮤지엄 RAW/SILVER/GOLD)는 별도 구글 드라이브 폴더로 관리한다. (링크 TODO —
RAG 담당자가 드라이브에 올린 뒤 여기에 추가)

새 데이터셋이 추가되면 로컬 정리 후 해당 드라이브 폴더에도 동일하게 업로드한다.

## 데이터셋 분할

현재 학습 대상은 `metadata.csv`에 등록된 7개 유물 클래스다. 원본 이미지는 수정하지 않고,
다음 명령으로 클래스별 70/15/15 비율의 학습/검증/테스트 데이터를 생성한다.

```powershell
python scripts/split_dataset.py
```

분할은 고정 seed(`42`)를 사용하므로 동일한 원본에서는 언제나 같은 결과가 생성된다.
PNG/JPG/JPEG/WebP 이미지만 복사하며 렌더링 로그 CSV는 제외한다.

## Vision 학습 환경

Python 3.10 이상을 사용하고, 의존성 관리는 [uv](https://docs.astral.sh/uv/)로 한다
(`pip install -r ...` 대신 `uv sync`). `pyproject.toml`에 CPU용/GPU용 PyTorch 인덱스가
`cpu`/`cu130` extra로 나뉘어 있으니, 노트북 종류에 맞는 extra로 동기화하면 된다.
`uv sync`가 `.venv`도 알아서 만들어준다.

### CPU 노트북 (Windows PowerShell)

```powershell
uv sync --extra cpu
.\.venv\Scripts\Activate.ps1
python scripts/check_vision_env.py
```

### NVIDIA GPU 노트북

먼저 `nvidia-smi`로 GPU와 드라이버를 확인한다.

```powershell
nvidia-smi
uv sync --extra cu130
.\.venv\Scripts\Activate.ps1
python scripts/check_vision_env.py
```

`cu130`은 CUDA 13.0 빌드다. 드라이버가 지원하는 CUDA 메이저 버전이 다르면
[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 맞는 인덱스 URL을 확인하고,
`pyproject.toml`의 `[[tool.uv.index]]`에 같은 패턴으로 새 extra(예: `cu126`)를 추가한 뒤
`uv sync --extra cu126`처럼 동기화한다.

점검 결과의 `selected_device`가 CPU 노트북에서는 `cpu`, GPU 노트북에서는 `cuda`여야 한다.
GPU 노트북에 `cpu` extra로 동기화하면 GPU가 있어도 `cpu`로 표시되므로 학습 전에 반드시 확인한다.

가상환경을 매번 activate하지 않고 실행하려면 `python ...` 대신 `uv run python ...`을 써도 된다.

### 데이터 로더와 모델 점검

다음 명령은 7개 클래스의 `ImageFolder` 매핑을 확인하고, ImageNet 사전학습 EfficientNet-B0에
train 데이터 한 배치를 통과시킨다. 최초 실행 시 사전학습 가중치 약 21MB를 `.cache/torch/`에 받는다.

```powershell
python scripts/check_vision_pipeline.py
```

네트워크 없이 데이터와 모델 구조만 확인하려면 `--no-pretrained` 옵션을 사용한다.

### 1단계 학습: classifier head

ImageNet 사전학습 backbone을 고정하고 새 6클래스 classifier만 학습한다. best checkpoint는
validation loss를 기준으로 `runs/vision/<run-name>/best_model.pth`에 저장되며, epoch별 기록은
같은 폴더의 `history.json`에 저장된다.

```powershell
# 전체 classifier 학습 (기본 5 epoch)
python -m ai.vision.train --run-name classifier_head

# GPU 노트북 예시
python -m ai.vision.train --run-name classifier_head_gpu --batch-size 32 --num-workers 4
```

다음 명령은 학습 파이프라인만 빠르게 확인하는 CPU smoke test다. 각 split의 일부 배치만 사용하므로
출력되는 loss와 accuracy는 모델 성능으로 해석하지 않는다.

```powershell
python -m ai.vision.train --epochs 1 --batch-size 4 `
  --max-train-batches 2 --max-val-batches 2 `
  --run-name smoke_classifier_cpu --device cpu
```

### 2단계 학습: fine-tuning

classifier 학습의 best checkpoint를 불러온 뒤 EfficientNet의 마지막 feature block 2개를 풀어
더 낮은 학습률로 미세 조정한다.

```powershell
python -m ai.vision.finetune `
  --checkpoint runs\vision\classifier_head\best_model.pth `
  --run-name fine_tune
```

GPU 노트북에서 실제 학습을 실행하는 전체 순서는 `Tomorrowdo.md`를 참고한다.

### 모델 평가

checkpoint를 synthetic test 전체에서 평가하고 accuracy, macro F1, 클래스별 지표,
confusion matrix와 오분류 이미지를 저장한다.

```powershell
python -m ai.vision.evaluate `
  --checkpoint runs\vision\fine_tune\best_model.pth `
  --split test `
  --run-name fine_tune_test
```

결과는 `runs/vision/evaluations/<run-name>/`에 생성된다. `--max-batches`를 사용한 결과는
일부 클래스만 포함할 수 있으므로 smoke test 용도로만 사용한다.

### 단일 이미지 추론

checkpoint와 이미지 한 장을 입력하면 최상위 예측과 top-k를 JSON으로 반환한다.

```powershell
python -m ai.vision.predict `
  --checkpoint runs\vision\fine_tune\best_model.pth `
  --image path\to\artifact.jpg `
  --top-k 3
```

`--output runs/vision/predictions/result.json`을 추가하면 같은 결과를 UTF-8 JSON 파일로도 저장한다.
추론 로직은 `ArtifactPredictor` 클래스로 분리되어 이후 FastAPI에서도 동일하게 재사용한다.

## 분류 API 실행

`uv sync`(위 단계)로 backend 의존성까지 이미 설치되어 있다. 사용할 checkpoint와 장치를
환경변수로 지정한다.

```powershell
$env:VISION_MODEL_PATH = "runs\vision\fine_tune_gpu\best_model.pth"
$env:VISION_DEVICE = "cuda"
$env:VISION_CONFIDENCE_THRESHOLD = "0.60"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

CPU에서는 `VISION_DEVICE=cpu`를 사용한다. 서버가 실행되면 Swagger UI는
`http://127.0.0.1:8000/docs`, 분류 API는 `POST /api/v1/classify`에서 확인한다.

```powershell
curl.exe -X POST -F "image=@path\to\artifact.jpg" `
  http://127.0.0.1:8000/api/v1/classify
```

기본 허용 프론트엔드 주소는 `http://localhost:5173`, `http://localhost:3000`이다.
다른 주소는 쉼표로 구분한 `CORS_ALLOWED_ORIGINS` 환경변수로 설정한다.

## 소장품 RAG (ai/rag)

GOLD 데이터(`data/gold/emuseum/`)의 카드(`context_text`)만 근거로 답한다.

**기본은 룰(ID 조회), RAG 검색은 채팅 보조.** Vision이 `artifact_id`를 정해 주므로 설명은 검색 없이 카드를 바로 꺼낸다.
채팅(`ai/rag/docent.py`)은 질문을 아래 순서로 보낸다.

| 순서 | 질문 | 처리 |
|---|---|---|
| 1 | 전체 수량·통계 ("박물관에 불상이 몇 점?") | 고정 안내 + 관련 예시 |
| 2 | 소장품 번호 ("신수 1846은?") | 번호 색인 조회 |
| 3 | 다른 소장품 이름 ("농경문 청동기도 알려줘") | 이름 사전 조회 (동명 4건 이상이면 검색) |
| 4 | 탐색 ("비슷한 유물", "같은 시대 다른 유물") | **RAG 검색** 결과를 LLM 없이 목록으로 답함. "같은 ○○"는 현재 유물 속성으로 필터 |
| 4-1 | 연관 ("관련된 소장품", "연관 유물") | 카드에 기록된 연관 소장품 목록 |
| 5 | 그 외 (기본) | 현재 유물 카드만 LLM에 전달 (연관 소장품은 카드 안의 이름·번호만) |

- 검색은 Parent-Child 구조다. child 청크(4,269개)를 벡터(Chroma) + BM25로 찾아 RRF로 합치고, LLM에는 parent 카드 전체를 넘긴다.
- 답변 문장 끝의 `[소장품 번호]`는 넘긴 카드의 번호만 남기고, 출처는 인용된 카드의 citation이다.

| 구성 | 선택 |
|---|---|
| 임베딩 | BGE-M3 (Ollama `bge-m3`) |
| 벡터 DB | Chroma (로컬, `data/index/emuseum/chroma`) |
| 키워드 검색 | BM25 + Kiwi 형태소, 한자는 글자 단위 토큰 추가 |
| 답변 LLM | EXAONE 3.5 7.8B (Ollama `exaone3.5:7.8b`) |

### 준비

```bash
ollama pull bge-m3
ollama pull exaone3.5:7.8b
uv pip install -r requirements-rag.txt   # pyproject.toml 반영 전 임시
python -m ai.rag.index                   # data/gold/ 가 있을 때 색인 1회 (약 2분)
```

### 사용

```bash
python -m ai.rag.docent bon002789 --describe                      # 유물 설명
python -m ai.rag.docent bon002789 "비슷한 유물 또 있어?" --no-llm  # 채팅 라우팅 확인
python scripts/check_docent_routes.py                             # 라우팅 점검 (LLM 없음)
python -m ai.rag.ask "신수 1846은 무엇인가요?"                     # 검색 단독
python -m ai.rag.evaluate                                         # 평가 (data/eval/emuseum_questions.jsonl)
```

## 통합 실행 (Vision + RAG + Backend + Frontend)

`feature/docent-rag-integration`은 `feature/rag-llm-kang`(Vision v2·도슨트 API)에
`feature/scan-flow`(프론트 시작화면·실시간 인식)와 소장품 RAG를 합친 브랜치다.

| 화면 | 동작 |
|---|---|
| 유물 설명 | 사전 생성본(`data/rag/gold/emuseum/descriptions_by_type.jsonl`) 우선, 없으면 RAG로 즉석 생성 |
| 채팅 | RAG (`ai/rag/docent.py`) |

Git에 없는 파일 (공유 드라이브에서 받아 아래 경로에 둔다):

```text
data/gold/emuseum/                          # RAG GOLD 카드·청크
data/index/emuseum/                         # 검색 색인 (없으면 python -m ai.rag.index 로 생성)
data/rag/gold/emuseum/descriptions_by_type.jsonl
runs/vision/fine_tune_gpu_v2/best_model.pth # Vision checkpoint
```

```bash
# 1) Backend (Ollama 실행 중이어야 함)
VISION_DEVICE=cpu python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 2) Frontend (실제 API 사용)
cd frontend && npm install
VITE_USE_MOCK=false npm run dev
```

프론트는 `http://localhost:5173`으로 연다. `127.0.0.1:5173`으로 열면
`CORS_ALLOWED_ORIGINS`에 그 주소를 추가해야 한다.
