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

## 데이터셋 분할

현재 학습 대상은 `metadata.csv`에 등록된 7개 유물 클래스다. 원본 이미지는 수정하지 않고,
다음 명령으로 클래스별 70/15/15 비율의 학습/검증/테스트 데이터를 생성한다.

```powershell
python scripts/split_dataset.py
```

분할은 고정 seed(`42`)를 사용하므로 동일한 원본에서는 언제나 같은 결과가 생성된다.
PNG/JPG/JPEG/WebP 이미지만 복사하며 렌더링 로그 CSV는 제외한다.

## Vision 학습 환경

Python 3.10 이상을 사용한다. CPU 노트북과 NVIDIA GPU 노트북은 동일한 학습 코드를 사용하지만,
PyTorch wheel이 다르므로 가상환경에서 PyTorch를 먼저 설치한 후 공통 의존성을 설치한다.

### CPU 노트북 (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-vision.txt
python scripts/check_vision_env.py
```

### NVIDIA GPU 노트북

먼저 `nvidia-smi`로 GPU와 드라이버를 확인한다. 그다음
[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 해당 운영체제와 지원되는
CUDA 버전에 맞는 명령을 선택해 `torch torchvision`을 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 이 위치에서 PyTorch 공식 설치 선택기가 안내한 CUDA 명령 실행
python -m pip install -r requirements-vision.txt
python scripts/check_vision_env.py
```

점검 결과의 `selected_device`가 CPU 노트북에서는 `cpu`, GPU 노트북에서는 `cuda`여야 한다.
GPU 노트북에 일반 PyPI/CPU 빌드가 설치되면 GPU가 있어도 `cpu`로 표시되므로 학습 전에 반드시 확인한다.

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

backend 의존성을 설치하고 사용할 checkpoint와 장치를 환경변수로 지정한다.

```powershell
python -m pip install -r requirements-backend.txt
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
