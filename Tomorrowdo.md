# GPU 학습 인수인계 및 다음 작업

## 0. 코드 보관 상태

Git 협업 규칙에 따라 최신 `origin/develop`에서 Vision 기능 브랜치를 만들고 원격에 보관한다.
`master`와 `develop`에는 직접 push하지 않는다.

- [x] 최신 `origin/develop` 확인
- [x] `feature/vision-efficientnet` 브랜치 생성
- [x] 현재 변경 파일 검증 및 commit
- [x] `origin/feature/vision-efficientnet` push

GPU 노트북에서는 아래 3절의 명령으로 이 feature 브랜치를 직접 clone한다.

## 1. 현재 완료된 상태

- [x] 6개 유물 클래스 메타데이터 등록
- [x] synthetic 이미지 1,170장 분할
- [x] `ImageFolder` train/val/test 로더 구현
- [x] ImageNet 사전학습 EfficientNet-B0 및 6클래스 classifier 구현
- [x] classifier-only 학습, validation, early stopping, best checkpoint 저장 구현
- [x] 마지막 feature block 2개 fine-tuning 구현
- [x] CPU smoke test로 forward/backward/checkpoint 저장 검증
- [ ] GPU에서 classifier 전체 학습
- [ ] GPU에서 fine-tuning 전체 학습
- [x] test 평가, confusion matrix, 오분류 저장 구현 및 CPU smoke test
- [ ] GPU 학습 모델의 전체 synthetic test 평가
- [x] 단일 이미지 top-k 추론 및 JSON 출력 구현, CPU smoke test
- [x] FastAPI `/api/v1/classify` 연결 및 CPU 계약 smoke test
- [ ] GPU 최종 checkpoint로 FastAPI 실서버 확인
- [ ] 실제 핸드폰 촬영 데이터 수집 및 평가

현재 CPU에서 실행한 학습은 train/val의 일부 배치만 사용한 **동작 확인용 smoke test**다.
해당 loss와 accuracy는 모델 성능으로 사용하지 않는다.

## 2. 데이터 현황

| artifact_id | 유물명 | train | val | test |
|---|---|---:|---:|---:|
| `bon002789` | 금동 반가사유상 | 140 | 30 | 30 |
| `duk000798` | 청동촛대 | 140 | 30 | 30 |
| `jub002084` | 백자 달항아리 | 119 | 26 | 25 |
| `ssu001794` | 농경문 청동기 | 140 | 30 | 30 |
| `ssu001846` | 방패형 동기 | 140 | 30 | 30 |
| `ssu003094` | 요령식 동검 | 140 | 30 | 30 |
| **합계** |  | **819** | **176** | **175** |

현재 데이터는 모두 Blender synthetic 이미지다. 현재 test 정확도는 synthetic 기준선이며,
실제 핸드폰 사진에 대한 서비스 성능을 의미하지 않는다.

## 3. GPU 노트북으로 옮길 것

Git에서 제외되는 항목이 있으므로 저장소 코드만 받아서는 학습할 수 없다.

1. `feature/vision-efficientnet`을 push한 뒤 GPU 노트북에서 해당 브랜치를 받는다.

```powershell
git clone --branch feature/vision-efficientnet `
  https://github.com/kgj200401/gg_minipjt_2.git
```

이미 저장소가 있다면 `git fetch origin` 후 `git switch feature/vision-efficientnet`을 사용한다.

2. 다음 데이터도 별도로 복사한다.
   - 필수: `data/raw/synthetic/`
   - 필수: `data/metadata.csv`
   - 선택: 이미 분할된 `data/processed/`
3. `data/processed/`를 복사하지 않았다면 GPU 노트북에서 다음 명령으로 동일하게 재생성한다.

```powershell
python scripts/split_dataset.py
```

분할 seed가 42로 고정되어 있어 같은 raw 데이터에서는 동일한 분할이 생성된다.
`.venv/`, `.cache/`, `runs/`는 복사하지 말고 각 노트북에서 새로 만든다.

## 4. GPU 환경 설치

GPU 노트북에서 PowerShell을 열고 프로젝트 루트에서 실행한다.

```powershell
nvidia-smi
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Python은 3.10 이상을 사용한다. 그다음 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
GPU 노트북의 운영체제와 드라이버가 지원하는 CUDA 버전을 선택하고, 표시된
`torch torchvision` 설치 명령을 실행한다. CPU용 wheel을 설치하면 GPU가 사용되지 않는다.

PyTorch 설치 후 나머지 패키지를 설치하고 환경을 확인한다.

```powershell
python -m pip install -r requirements-vision.txt
python scripts/check_vision_env.py
```

반드시 다음과 비슷하게 나와야 한다.

```text
cuda_available=True
selected_device=cuda
cuda_device_0=<GPU 이름>
efficientnet_b0_smoke_test=ok
```

`selected_device=cpu`가 나오면 학습을 시작하지 말고 PyTorch CUDA wheel과 NVIDIA 드라이버부터 확인한다.

## 5. 데이터·모델 파이프라인 확인

```powershell
python scripts/check_vision_pipeline.py
```

확인할 값:

```text
selected_device=cuda
train_images=819
val_images=176
test_images=175
batch_images_shape=(4, 3, 224, 224)
logits_shape=(4, 6)
vision_pipeline_smoke_test=ok
```

## 6. GPU 본 학습 1단계: classifier

ImageNet backbone을 고정하고 classifier만 전체 데이터로 학습한다.

```powershell
python -m ai.vision.train `
  --device cuda `
  --epochs 5 `
  --batch-size 32 `
  --num-workers 4 `
  --run-name classifier_head_gpu
```

GPU 메모리가 부족하면 `--batch-size 16`으로 낮춘다. Windows에서 worker 관련 오류가 발생하면
`--num-workers 0`으로 다시 실행한다.

생성 결과:

```text
runs/vision/classifier_head_gpu/best_model.pth
runs/vision/classifier_head_gpu/history.json
```

## 7. GPU 본 학습 2단계: fine-tuning

classifier best checkpoint를 시작점으로 사용한다. 마지막 feature block 2개는 `1e-5`,
classifier는 `1e-4` 학습률을 사용하며 validation loss가 개선되지 않으면 early stopping한다.

```powershell
python -m ai.vision.finetune `
  --checkpoint runs\vision\classifier_head_gpu\best_model.pth `
  --device cuda `
  --epochs 15 `
  --batch-size 16 `
  --num-workers 4 `
  --feature-blocks 2 `
  --run-name fine_tune_gpu
```

생성 결과:

```text
runs/vision/fine_tune_gpu/best_model.pth
runs/vision/fine_tune_gpu/history.json
```

두 단계 모두 실행 로그에서 전체 train/val 표본 수가 처리되는지 확인한다. `--max-train-batches`와
`--max-val-batches`는 smoke test 전용이므로 GPU 본 학습에는 사용하지 않는다.

## 8. 학습 결과 보관

`runs/`와 `*.pth`는 Git에서 제외된다. GPU 학습이 끝나면 다음 파일을 별도 저장소나 외장 저장장치로
현재 프로젝트에 다시 가져와야 한다.

- `runs/vision/classifier_head_gpu/history.json`
- `runs/vision/classifier_head_gpu/best_model.pth`
- `runs/vision/fine_tune_gpu/history.json`
- `runs/vision/fine_tune_gpu/best_model.pth`
- `runs/vision/evaluations/classifier_head_gpu_test/` 전체
- `runs/vision/evaluations/fine_tune_gpu_test/` 전체
- 필요한 경우 `runs/vision/predictions/` 결과

최종 후보는 `fine_tune_gpu/best_model.pth`지만, 다음 단계에서 classifier 모델과 fine-tuned 모델을
동일한 test 데이터로 비교한 뒤 확정한다.

## 9. GPU 모델 전체 평가

classifier 모델과 fine-tuned 모델을 동일한 synthetic test 175장에서 각각 평가한다.
GPU 본 평가에는 `--max-batches`를 사용하지 않는다.

```powershell
python -m ai.vision.evaluate `
  --checkpoint runs\vision\classifier_head_gpu\best_model.pth `
  --split test `
  --device cuda `
  --batch-size 32 `
  --num-workers 4 `
  --run-name classifier_head_gpu_test

python -m ai.vision.evaluate `
  --checkpoint runs\vision\fine_tune_gpu\best_model.pth `
  --split test `
  --device cuda `
  --batch-size 32 `
  --num-workers 4 `
  --run-name fine_tune_gpu_test
```

각 평가 폴더에는 다음 결과가 생성된다.

```text
summary.json              # accuracy, macro F1, 오분류 수
per_class_metrics.csv     # 클래스별 precision/recall/F1/support
confusion_matrix.png
misclassified.csv
misclassified/            # 오분류 원본 이미지 복사본
```

두 `summary.json`과 confusion matrix를 비교해 최종 후보를 결정한다. 현재 test는 모두 synthetic이므로
이 결과만으로 실제 서비스 성능을 확정하지 않는다.

## 10. 최종 후보 단일 이미지 추론

평가 결과가 더 좋은 checkpoint를 선택한 뒤 synthetic 이미지와 별도로 준비한 이미지에서 JSON 출력을 확인한다.

```powershell
python -m ai.vision.predict `
  --checkpoint runs\vision\fine_tune_gpu\best_model.pth `
  --image path\to\artifact.jpg `
  --top-k 3 `
  --device cuda `
  --output runs\vision\predictions\artifact_result.json
```

주요 출력 필드:

```json
{
  "artifact_id": "ssu001794",
  "artifact_name": "농경문 청동기",
  "confidence": 0.94,
  "top_k": []
}
```

confidence가 낮거나 top-1과 top-2 차이가 작으면 모델이 확신하지 못한 입력으로 처리하는 정책이 필요하다.
실제 confidence 임계값은 real validation 데이터로 정하며 synthetic 결과만 보고 결정하지 않는다.

## 11. GPU 최종 모델로 FastAPI 실행

backend 패키지를 설치한 뒤 최종 checkpoint와 CUDA 장치를 지정한다.

```powershell
python -m pip install -r requirements-backend.txt
$env:VISION_MODEL_PATH = "runs\vision\fine_tune_gpu\best_model.pth"
$env:VISION_DEVICE = "cuda"
$env:VISION_CONFIDENCE_THRESHOLD = "0.60"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000/docs`를 열어 이미지 업로드를 시험하거나 다음 명령을 사용한다.

```powershell
curl.exe -X POST -F "image=@path\to\artifact.jpg" `
  http://127.0.0.1:8000/api/v1/classify
```

분류 API는 다음을 처리한다.

- 서버 시작 시 checkpoint 한 번만 로드
- JPG/PNG MIME 및 실제 디코딩 형식 확인
- 기본 10MB, 2,500만 픽셀 제한
- top-3 후보 반환
- confidence가 임계값보다 낮으면 재촬영 안내
- `localhost:5173`, `localhost:3000` CORS 허용

`VISION_CONFIDENCE_THRESHOLD=0.60`은 임시값이다. 실제 촬영 validation 데이터의 오인식과
미인식 비율을 측정한 후 결정한다.

## 12. 이어서 할 작업

1. 실제 핸드폰 사진을 촬영 세션 기준으로 val/test에 분리
2. synthetic와 real 성능을 별도로 평가
3. real validation 결과로 confidence 임계값 확정
4. 프론트엔드 업로드 화면을 `/api/v1/classify`에 연결
5. 분류된 `artifact_id`를 RAG 설명 API로 연결
