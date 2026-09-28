# RAG 실험 사용 설명서

현재 코드 기준으로 **Gold 생성 → YAML 설정 → Vector DB 구축 → Retriever 평가 → 모델 답변 생성**을 순서대로 설명합니다.

모든 명령은 `pyproject.toml`이 있는 프로젝트 루트에서 실행합니다. YAML 예시는 기존 파일의 해당 항목에 합쳐 넣거나 같은 항목을 수정해서 사용합니다. 출력 예시는 사용 방법을 설명하기 위한 것이며 실제 성능이나 모델 답변을 보장하지 않습니다.

## 시작하기

```powershell
cd C:\Users\Admin\llm_project
uv sync
```

Python 3.12 이상을 사용합니다. `uv sync`는 프로젝트 의존성과 실행 환경을 준비합니다. Hugging Face 모델을 직접 로딩하거나 양자화할 때는 추가 의존성도 설치합니다.

```powershell
uv sync --extra hf
```

이후 Transformers 실행 명령에는 `uv run --extra hf`를 사용합니다. 현재 Transformers 로더는 CUDA GPU를 요구하며 CPU 실행 경로는 구현되어 있지 않습니다. 임베딩은 CPU에서도 가능합니다. 양자화 방식별 실제 지원 여부는 GPU와 설치된 라이브러리에 따라 달라집니다.

```text
ai/
├─ rag/
│  ├─ storage/       # 원본 3D 데이터 및 experiments/의 Chroma DB
│  ├─ tests/         # 구현 검증용 테스트
│  ├─ prompt/
│  ├─ scripts/       # Gold 생성, 검색 확인, 모델 준비
│  └─ *.py           # 설정 로딩·문서화·청킹·임베딩·검색
└─ llm/
   ├─ configs/       # base.yaml 및 experiments/*.yaml
   ├─ model/         # 저장한 모델
   ├─ evaluation/    # 평가 코드, evalset/, results/
   └─ *.py           # 모델 로딩·RAG 답변 생성
```

`storage/`, `model/`, `evaluation/results/`는 `.gitkeep`만 Git에 포함됩니다. 현재 `.gitignore`는 `data/`도 제외하므로 새 환경에서는 Silver/Gold 데이터를 별도로 준비합니다. 아래 파이프라인은 텍스트 Gold를 사용하며 3D 파일을 읽지 않습니다.

## 1. Gold 데이터 생성

### 1-1. 입력 파일 준비

| 파일 | 역할 |
|---|---|
| `data/silver/Silver_0928.jsonl` | 유물 본체 정보 |
| `data/silver/Silver_images_0928.jsonl` | 유물별 이미지 정보 |
| `data/silver/Silver_relations_0928.jsonl` | 유물 사이의 관계 정보 |
| `data/silver/var_list.md` | Gold에 포함할 본체 필드 목록. 표 첫 열의 백틱 변수명을 읽음 |

입력 경로는 `ai/rag/scripts/build_gold_dataset.py`에 고정되어 있습니다. YAML의 `data.gold_file`은 생성된 Gold를 읽을 경로이며 Silver 입력을 바꾸는 설정이 아닙니다.

스크립트의 `TARGET_IDS`에 있는 다음 소장품번호(`relic_label`)를 선택합니다.

```text
본관 2789, 덕수 798, 접수 702, 신수 1794,
신수 1846, 신수 3094, 신수 22891
```

대상을 바꾸려면 `TARGET_IDS`를 수정합니다. 앞뒤 공백만 제거하여 비교하며 내부 공백은 유지합니다. 선택 후에는 내부 `relic_id`로 이미지와 관계를 결합합니다. YAML의 `rag.relic_label`은 Gold 생성 대상을 결정하지 않습니다.

### 1-2. 생성 명령 실행

처음 생성할 때:

```powershell
uv run python ai/rag/scripts/build_gold_dataset.py
```

기존 Gold 파일을 교체할 때:

```powershell
uv run python ai/rag/scripts/build_gold_dataset.py --overwrite
```

다른 이름으로 저장할 때:

```powershell
uv run python ai/rag/scripts/build_gold_dataset.py --output data/gold/Gold_custom.jsonl
```

`--overwrite` 없이 출력 파일이 이미 존재하면 중단합니다. 출력 이름을 바꾸면 이후 YAML의 `data.gold_file`도 같은 경로로 변경합니다.

### 1-3. 결과 확인

기본 출력은 `data/gold/Gold_0928.jsonl`입니다. 한 줄에 유물 하나를 담는 JSONL이며 본체 필드와 `images`, `relations` 배열이 저장됩니다.

콘솔 출력 예시(일부):

```text
Gold records: 7
Unique artifacts: 7
Duplicate artifact IDs: 0
Missing artifact IDs: 0
...
INFO:__main__:Gold 저장: .../data/gold/Gold_0928.jsonl
```

개수는 실제 입력에 따라 달라집니다. Gold는 검색용 원본 데이터이며, 4장에서 만드는 질문·정답 청크 목록인 **평가 정답셋**과는 별개입니다.

## 2. YAML 설정 요소

### 2-1. 설정 파일을 읽는 방식

`ai/rag/config.py`의 `load_config()`는 `ai/llm/configs/base.yaml`을 읽고 `--config`로 지정한 YAML의 값을 항목별로 덮어씁니다. 실험 파일에서 생략한 항목은 base 값을 상속합니다.

파일명은 자유롭게 정할 수 있지만 `.yaml`로 만들고 실행할 때 정확한 경로를 지정합니다. DB 폴더와 결과 파일명은 YAML 파일명이 아니라 `experiment.name`으로 결정됩니다.

실험 파일 예시 — `ai/llm/configs/experiments/exp01_recursive_500.yaml`:

```yaml
experiment:
  name: recursive_500_overlap50_bge_m3
chunking:
  strategy: recursive
  chunk_size: 500
  chunk_overlap: 50
embedding:
  model_name: BAAI/bge-m3
rag:
  relic_label: "본관 2789"
  smoke_query: null
evaluation:
  dataset: ai/llm/evaluation/evalset/eval_dataset_exp01.json
llm:
  enabled: false
```

프로젝트 파일 경로는 프로젝트 루트 기준 상대 경로 또는 절대 경로를 사용합니다. 모델 ID는 `BAAI/bge-m3` 같은 Hub 식별자일 수도 있습니다. 아래 숫자 범위는 정상 사용을 위한 범위이며 모든 잘못된 자료형이나 범위를 설정 로더가 사전에 검증하는 것은 아닙니다.

### 2-2. 데이터·실험·검색 대상

| 설정 | 입력 가능한 값 / 예시 | 의미 |
|---|---|---|
| `data.gold_file` | 존재하는 Gold JSONL 경로 | DB 구축과 소장품번호를 내부 ID로 변환할 때 사용 |
| `experiment.name` | 경로 구분자 없는 비어 있지 않은 이름. 예: `recursive_500_overlap50_bge_m3` | DB 폴더와 결과 파일명. `.`과 `..`는 불가 |
| `rag.relic_label` | Gold에 유일하게 존재하는 소장품번호 문자열. 예: `"본관 2789"` | 단일 검색·답변 생성 대상. DB 구축 실행에서도 유효한 값 필요 |
| `rag.smoke_query` | 질문 문자열 또는 `null` | DB 구축 후 간단한 검색 확인. null이나 빈 문자열이면 생략 |

`rag.relic_label`은 **DB에 넣을 유물 범위를 제한하지 않습니다.** DB는 전체 Gold 또는 `--sample`로 선택한 앞부분으로 구축하고 검색할 때 선택된 유물로 필터링합니다. 평가 대상은 각 평가 JSON 항목의 `relic_label`이며 YAML 값을 기본값으로 사용하지 않습니다. YAML의 `rag.relic_id`는 허용하지 않습니다.

### 2-3. 청킹

| 설정 | 입력 가능한 값 | 의미 |
|---|---|---|
| `chunking.strategy` | `recursive`, `sentence_based`, `field_based`, `none` | 아래 표의 분할 방법 |
| `chunking.chunk_size` | 양의 정수. 예: `500`, `800` | 청크 크기 기준. 토큰 수가 아닌 **문자 수** |
| `chunking.chunk_overlap` | 0 이상, chunk_size 미만의 정수 | 이웃 청크에 중복시킬 문자 수. 예: `50` |

| strategy | 동작 |
|---|---|
| `recursive` | 구분자를 이용해 텍스트를 지정 크기로 재귀 분할 |
| `sentence_based` | 문장부호·줄바꿈을 경계로 묶음. 긴 단일 문장은 크기 기준을 넘을 수 있음 |
| `field_based` | 기본 정보(basic), 상세 설명(detail), 관계(relations) 단위로 분할 |
| `none` | 유물 하나를 하나의 청크로 사용 |

field_based와 none은 크기·overlap으로 본문을 추가 분할하지 않습니다. 다만 공통 설정 검증 때문에 유효한 `chunk_size`, `chunk_overlap`은 필요합니다.

### 2-4. 임베딩과 Vector DB

| 설정 | 입력 가능한 값 / 예시 | 의미 |
|---|---|---|
| `embedding.model_name` | HuggingFaceEmbeddings로 로딩 가능한 모델 ID 또는 로컬 경로. 기본 `BAAI/bge-m3` | 문서·질문을 벡터로 만드는 모델. 답변 생성용 LLM과 별개 |
| `embedding.device` | `auto`, `cpu`, `cuda`, `cuda:0` 등 백엔드 지원 장치 문자열 | auto는 CUDA가 있으면 CUDA, 없으면 CPU |
| `embedding.normalize_embeddings` | `true`, `false` | 임베딩 정규화 여부 |
| `embedding.query_prefix` | 문자열. 없으면 `""` | 질문 앞에 붙일 텍스트. 모델이 요구하는 경우 사용 |
| `embedding.document_prefix` | 문자열. 없으면 `""` | 문서 앞에 붙일 텍스트 |
| `vectorstore.type` | `chroma`만 지원 | 저장소 종류 |
| `vectorstore.base_dir` | 디렉터리 경로. 기본 `ai/rag/storage/experiments` | 실험별 DB의 상위 폴더 |
| `vectorstore.collection_name` | Chroma가 허용하는 컬렉션 이름. 기본 `gold_artifacts` | DB 내부 컬렉션. 구축과 검색 시 동일하게 사용 |

DB 경로는 `{base_dir}/{experiment.name}/chroma/`입니다. 현재 YAML로 거리 함수를 변경하는 옵션은 없습니다. 기존 DB 검색 시 임베딩 모델·정규화·접두어·컬렉션 설정을 구축 때와 일치시키세요. Gold·청킹·문서 임베딩 설정을 바꾸면 새 실험 이름으로 DB를 구축하고 정답 청크도 다시 검토합니다.

### 2-5. Retriever와 평가

| 설정 | 입력 가능한 값 | 의미 |
|---|---|---|
| `retriever.search_type` | `similarity`, `mmr` | 유사도 검색 또는 유사도와 다양성을 함께 고려하는 MMR 검색 |
| `retriever.k` | 양의 정수. 기본 `5` | 일반 검색·RAG 답변에 사용할 최대 청크 수 |
| `retriever.fetch_k` | 양의 정수. 예: `20` | MMR 후보 수. 실제 호출은 최소 k 이상으로 보정. similarity에서는 미사용 |
| `retriever.lambda_mult` | `0.0`~`1.0` | MMR에서 1에 가까울수록 유사도, 0에 가까울수록 다양성 중시. similarity에서는 미사용 |
| `evaluation.dataset` | 평가 JSON 경로 | 질문별 소장품번호와 정답 청크 ID 목록 |
| `evaluation.metrics` | `chunk_recall_at_1`, `chunk_recall_at_3`, `chunk_recall_at_5`로 구성한 배열 | 기본값은 세 개 모두 |

현재 평가기는 **항상 최대 5개를 검색하고 Recall@1·3·5를 모두 계산**합니다. retriever.k나 evaluation.metrics를 바꿔도 평가 K나 출력 지표가 선택적으로 바뀌지는 않습니다. metrics는 현재 허용 이름 검사에만 사용됩니다. Precision, MRR, nDCG, LLM 답변 품질 지표는 구현되어 있지 않습니다.

### 2-6. LLM 공통 설정

| 설정 | 입력 가능한 값 | 의미 |
|---|---|---|
| `llm.enabled` | `true`, `false` | --ask로 답변 생성 시 true 필요. DB 구축·Retriever 평가는 LLM을 호출하지 않음 |
| `llm.provider` | `ollama`, `transformers` | Ollama 서버 호출 또는 Python에서 모델 직접 로딩 |
| `llm.model` | Ollama 태그 / Hugging Face ID / 로컬 폴더 | 예: `qwen3:8b-q4_K_M`, `Qwen/Qwen3-1.7B`, `ai/llm/model/qwen3-1.7b-nf4` |
| `llm.visitor_type` | `test`, `child`, `general`, `expert` | 답변 대상. 검색 순위에는 영향 없음 |
| `llm.temperature` | 0 이상의 유한한 숫자. 예: `0.0`, `0.7` | Transformers는 0일 때 샘플링을 끄고 양수이면 샘플링. Ollama에는 값 전달 |

| visitor_type | 답변 지시 |
|---|---|
| `test` | 대상별 추가 말투·난이도 지시 없음. 기본 근거 기반 한국어 답변 지시는 유지 |
| `child` | 쉬운 어휘와 짧은 문장, 유치원 선생님 같은 말투 |
| `general` | 일반 성인에게 박물관 도슨트 수준으로 설명 |
| `expert` | 자료가 뒷받침하는 범위에서 전문 용어·제작 기법·역사 맥락을 상세히 설명 |

실제 시스템 프롬프트는 `ai/llm/rag_chain.py`의 `BASE_INSTRUCTION`, `VISITOR_INSTRUCTIONS`에 있습니다. `ai/rag/prompt/` 문서를 수정해도 런타임 프롬프트가 자동 변경되지는 않습니다.

### 2-7. Transformers 전용 설정

다음 항목은 현재 Ollama 호출에는 전달되지 않습니다.

| 설정 | 입력 가능한 값 | 의미 |
|---|---|---|
| `llm.source` | `huggingface`, `local` | Hub 원본 또는 저장 폴더에서 오프라인 로딩 |
| `llm.quantization` | `nf4`, `int4`, `int8`, `fp8`, `bf16`, `fp16`, `none` | 아래 표 참고. YAML은 `4bit`→nf4, `8bit`→int8 별칭도 지원 |
| `llm.device` | `cuda:0`, `cuda:1` 등 cuda:숫자 | 모델 전체를 올릴 단일 GPU. cpu·auto 미지원 |
| `llm.compute_dtype` | `auto`, `float16`, `bfloat16` | 로딩 시 계산 dtype. auto는 선택 GPU의 BF16 지원에 따라 BF16 또는 FP16 |
| `llm.trust_remote_code` | `true`, `false` | 모델 저장소의 사용자 정의 코드 허용 여부. 기본 false |
| `llm.max_input_tokens` | 양의 정수. 기본 `4096` | 템플릿·검색 근거·질문의 입력 제한. 초과 시 자동 절단 없이 오류 |
| `llm.max_new_tokens` | 양의 정수. 기본 `512` | 최대 생성 토큰 수. 입력과 합쳐 모델 컨텍스트 한도도 검사 |
| `llm.enable_thinking` | `true`, `false` | 채팅 템플릿에 전달하는 옵션. 효과는 템플릿 지원에 따름 |
| `llm.save_dir` | 저장 폴더 경로 | prepare_llm.py 전용. `{quantization}`을 방식명으로 치환. --output이 우선 |

| quantization | 현재 코드의 처리 |
|---|---|
| `nf4` | bitsandbytes 4비트 NF4 + double quantization |
| `int4` | torchao Int4WeightOnlyConfig(group_size=128)로 가중치 INT4 |
| `int8` | bitsandbytes load_in_8bit=True |
| `fp8` | torchao Float8WeightOnlyConfig로 가중치 FP8 E4M3FN |
| `bf16` | BF16 dtype으로 로딩. 4·8비트 압축 양자화가 아닌 정밀도 변환 |
| `fp16` | FP16 dtype으로 로딩. 정밀도 변환 |
| `none` | 추가 양자화 없이 모델 설정의 dtype을 auto로 로딩. 반드시 FP32라는 뜻은 아님 |

bf16·fp16은 해당 dtype이 우선합니다. none과 source: local은 저장된 dtype을 사용하므로 compute_dtype으로 덮어쓰지 않습니다. compute_dtype은 모든 양자화 방식의 내부 연산 dtype을 일괄 강제하는 옵션이 아닙니다.

`source: local`은 저장된 config.json의 양자화 설정을 읽으며 재양자화하지 않습니다. YAML의 quantization은 허용된 값이어야 하지만 실제 방식 선택에는 사용하지 않습니다. `source: huggingface`에서 이미 양자화된 Hub 모델은 거부하므로 직접 양자화하려면 원본 모델 ID를 사용합니다.

저장 경로 예시:

```yaml
llm:
  save_dir: "ai/llm/model/qwen3-1.7b-{quantization}"
```

`{quantization}`만 치환됩니다. `{nf4}`라고 쓰면 중괄호까지 폴더명에 남습니다. 기존 exp01 설정이 `{nf4}`라면 위 형식으로 고치거나 5장의 --output 예시를 사용하세요.

## 3. YAML대로 Vector DB 구축

### 3-1. 평가 정답셋이 아직 없을 때 준비

Gold 생성 후 exp01 YAML에서 data.gold_file, experiment.name, 청킹·임베딩 설정을 확인합니다. 정답 청크 ID는 청킹 후에 알 수 있으므로 DB와 청크 목록부터 만듭니다.

현재 run_experiment.py에는 --skip-evaluation 옵션이 없습니다. 평가 JSON이 있으면 자동 평가하므로 작성 중인 JSON 대신 **실제로 존재하지 않는 경로**를 임시 지정합니다. 그 파일을 새로 만들지는 마세요.

```yaml
evaluation:
  dataset: ai/llm/evaluation/evalset/not_created_exp01.json
rag:
  relic_label: "본관 2789"
  smoke_query: "이 유물의 재질은 무엇인가요?"
```

이 경로에 파일이 이미 있다면 다른 미사용 이름을 지정합니다. 빈 배열 `[]`은 평가 오류를 내므로 평가 생략 용도로 사용하지 않습니다. 작성 중인 기존 정답셋은 그대로 보관합니다.

### 3-2. 구축 실행

DB가 없을 때:

```powershell
uv run python ai/llm/evaluation/run_experiment.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml
```

동일 실험의 기존 DB를 교체할 때만 --rebuild를 붙입니다.

```powershell
uv run python ai/llm/evaluation/run_experiment.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --rebuild
```

실행 순서는 다음과 같습니다.

```text
Gold 읽기 → 문서 생성 → 청킹 → 임베딩 → Chroma 저장
→ 청크 목록 저장 → 정답셋이 있으면 평가
→ smoke_query가 있으면 검색 → 결과 저장
```

첫 실행은 임베딩 모델 다운로드와 임베딩 계산으로 오래 걸릴 수 있습니다. llm.enabled가 true여도 여기서는 답변 생성 모델을 로딩하지 않습니다.

### 3-3. 생성 파일과 출력

```text
ai/rag/storage/experiments/recursive_500_overlap50_bge_m3/chroma/
├─ chroma.sqlite3
└─ <벡터 세그먼트 UUID>/
   └─ *.bin

ai/llm/evaluation/results/
├─ recursive_500_overlap50_bge_m3_chunks.jsonl
└─ recursive_500_overlap50_bge_m3.json
```

SQLite와 .bin은 Chroma가 관리합니다. 사람이 정답 청크를 고를 때는 *_chunks.jsonl을 읽습니다. 결과 JSON은 콘솔에도 출력됩니다. 평가를 생략한 경우의 축약 예시:

```json
{
  "experiment": "recursive_500_overlap50_bge_m3",
  "gold_records": 7,
  "chunks": 17,
  "relic_id": "nmm-bon-002789-00",
  "relic_label": "본관 2789",
  "gold_file": "data/gold/Gold_0928.jsonl",
  "chunk_manifest": "ai/llm/evaluation/results/recursive_500_overlap50_bge_m3_chunks.jsonl",
  "smoke_query": "이 유물의 재질은 무엇인가요?"
}
```

7개 유물·17개 청크는 현재 저장된 결과 기준이며 설정과 입력에 따라 달라집니다. 실제 smoke 결과에는 smoke_ids, smoke_chunk_ids도 포함됩니다. smoke_query가 null이면 smoke 관련 필드는 생략됩니다. 정답 파일이 없으면 지표를 건너뛴다는 경고가 나오고 구축 결과를 저장합니다.

### 3-4. 선택적으로 소규모 구축

```powershell
uv run python ai/llm/evaluation/run_experiment.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --sample 2
```

Gold의 앞 2개만 사용하고 `{experiment.name}_sample2` 폴더·결과를 만듭니다. rag.relic_label은 그 2개 안에 있어야 합니다. 샘플 실행은 평가를 생략합니다. 해당 DB 검색 시에도 test_retriever.py에 --sample 2를 지정해야 합니다. evaluate_existing.py는 샘플 옵션을 지원하지 않으므로 정식 평가는 전체 DB로 진행합니다.

이미 유효한 DB와 청크 목록이 있다면 재구축하지 않고 4장으로 넘어갑니다.

## 4. YAML대로 Retriever 평가

### 4-1. 청크 목록에서 정답 고르기

`ai/llm/evaluation/results/recursive_500_overlap50_bge_m3_chunks.jsonl`을 엽니다. Gold에서 평가할 relic_label에 대응하는 relic_id를 찾고 청크 목록에서 그 ID의 page_content를 읽습니다.

실제 저장된 청크 예시:

```json
{
  "chunk_id": "nmm-bon-002789-00:recursive:recursive:567acf1381539f25353e035708610282251fb41bb7c24896b454bf261bd6eaf0",
  "relic_id": "nmm-bon-002789-00",
  "chunk_index": 1,
  "chunk_type": "recursive",
  "page_content": "크기: 높이 81.5cm, 불신높이 50cm\n지정 종류: 국보\n지정 번호: 국보 78호"
}
```

“이 불상의 전체 높이는 얼마인가요?”의 근거가 있으므로 이 청크를 정답으로 지정할 수 있습니다. 근거 청크가 여러 개면 모두 고릅니다. 검색 결과를 그대로 정답으로 삼지 말고 본문을 읽어 결정합니다. SQLite의 숫자 행 ID가 아닌 chunk_id 문자열 전체를 복사합니다.

### 4-2. 평가 JSON 작성

`ai/llm/evaluation/evalset/eval_dataset_exp01.json`에 다음 형식으로 작성합니다. JSONL이 아닌 **JSON 배열**입니다. 아래는 현재 청크 목록에 해당 ID가 있을 때 사용할 수 있는 1문항 예시입니다. 작성 중인 다른 질문을 덮어쓸 필요는 없습니다.

```json
[
  {
    "query": "이 불상의 전체 높이는 얼마인가요?",
    "relic_label": "본관 2789",
    "relevant_chunk_ids": [
      "nmm-bon-002789-00:recursive:recursive:567acf1381539f25353e035708610282251fb41bb7c24896b454bf261bd6eaf0"
    ]
  }
]
```

| 필드 | 작성 규칙 |
|---|---|
| query | 비어 있지 않은 실제 질문. 유물명을 반드시 넣을 필요 없음 |
| relic_label | Gold에서 유일하게 찾을 수 있는 소장품번호. 질문마다 지정 |
| relevant_chunk_ids | 해당 실험·유물의 근거 청크 ID 배열. 비어 있으면 안 되며 중복 금지 |

relic_id 입력은 거부됩니다. JSON에는 주석이나 마지막 항목 뒤 쉼표를 넣지 않습니다. 근거가 없는 질문은 이 Recall 정답셋에서 제외하거나 별도 평가 대상으로 관리합니다. 현재 형식은 빈 정답 배열을 허용하지 않습니다.

작성 중인 exp01 정답셋의 빈 질문·빈 소장품번호·`<...>` 임시 청크 ID를 완성하거나 미완성 항목을 제외하고 실행하세요. `접수 2084` 같은 번호도 Gold 포함 여부를 확인해야 합니다. 현재 생성 대상의 `접수 702`와 임의로 대체하면 안 됩니다.

코드는 JSON 형식과 Gold의 소장품번호는 검증하지만 **정답 chunk_id가 실제 DB에 존재하는지는 사전 검증하지 않습니다.** 잘못된 ID도 문자열이면 통과하여 Recall을 낮출 수 있으므로 실제 청크 목록과 대조하세요.

### 4-3. YAML 평가 경로 복원

3장의 임시 경로를 완성한 정답셋 경로로 바꿉니다.

```yaml
evaluation:
  dataset: ai/llm/evaluation/evalset/eval_dataset_exp01.json
  metrics: [chunk_recall_at_1, chunk_recall_at_3, chunk_recall_at_5]
```

### 4-4. 기존 DB로 평가 실행

```powershell
uv run python ai/llm/evaluation/evaluate_existing.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml
```

기존 DB에서 질문 임베딩으로 검색합니다. 문서 임베딩과 DB를 다시 구축하지 않고 LLM 답변도 생성하지 않습니다. 이번 실행만 별도 정답 파일을 지정할 수도 있습니다.

```powershell
uv run python ai/llm/evaluation/evaluate_existing.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --dataset ai/llm/evaluation/evalset/eval_dataset_exp01.json
```

### 4-5. 평가 결과 읽기

콘솔과 다음 파일에 결과를 저장합니다.

```text
ai/llm/evaluation/results/recursive_500_overlap50_bge_m3_retrieval_eval.json
```

다음은 **계산 설명용 가상 출력**입니다. A, B, C는 축약 ID이므로 실제 정답셋에 입력하면 안 됩니다.

```json
{
  "per_query": [
    {
      "query": "이 유물의 특징은 무엇인가요?",
      "relic_label": "본관 2789",
      "relic_id": "nmm-bon-002789-00",
      "relevant_chunk_ids": ["A", "B"],
      "retrieved_chunk_ids": ["A", "C", "B"],
      "chunk_recall_at_1": 0.5,
      "chunk_recall_at_3": 1.0,
      "chunk_recall_at_5": 1.0
    }
  ],
  "averages": {
    "chunk_recall_at_1": 0.5,
    "chunk_recall_at_3": 1.0,
    "chunk_recall_at_5": 1.0
  },
  "counts": {"queries": 1}
}
```

`Recall@K = 상위 K개 결과에 포함된 정답 청크 수 / 전체 정답 청크 수`입니다. 정답 2개 중 1위에는 A만 있으므로 Recall@1은 1/2입니다. averages는 질문별 점수의 산술평균이고 counts.queries는 평가 질문 수입니다.

이 평가는 **선택된 유물 내부에서 올바른 근거 청크를 찾는 성능**입니다. 전체 유물 중 어느 유물인지 식별하는 성능이나 답변 품질을 측정하지 않습니다. 청크가 5개 이하인 유물은 Top-5에 모두 포함되어 Recall@5가 쉽게 1이 될 수 있으므로 @1과 @3도 함께 봅니다.

`{experiment.name}.json`은 구축 시점 기록입니다. evaluate_existing.py는 이를 갱신하지 않고 별도의 `_retrieval_eval.json`을 씁니다. 과거 결과의 임시 정답이나 예전 smoke 질문을 새 실행 결과로 오해하지 마세요.

### 4-6. 여러 실험 비교

청킹을 바꾸면 각 실험의 청크 목록을 만들고 정답 ID를 다시 선정합니다. 질문은 유지할 수 있지만 정답 ID는 달라질 수 있습니다. YAML의 evaluation.dataset도 실험별로 지정하세요. 따로 지정하지 않은 실험은 base의 exp01 정답셋을 상속하므로 그대로 비교하면 안 됩니다.

모든 실험 설정과 정답셋을 준비한 뒤 전체 재구축·평가가 필요할 때:

```powershell
uv run python ai/llm/evaluation/run_all_experiments.py --rebuild
```

experiments/의 모든 *.yaml을 파일명 순서로 실행하고 실험별 DB·청크 목록·결과 JSON을 만듭니다. 통합 순위표는 생성하지 않으며 도중 오류가 나면 중단합니다. 단순 재평가는 각 YAML에 대해 evaluate_existing.py를 실행하세요.

## 5. YAML대로 모델 사용해 보기

### 5-1. 먼저 검색 확인

3장에서 만든 DB와 같은 실험·임베딩 설정으로 실행합니다.

```powershell
uv run python ai/rag/scripts/test_retriever.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --query "이 불상의 전체 높이는 얼마인가요?"
```

--ask가 없으므로 LLM을 사용하지 않습니다. 각 청크의 metadata와 본문 앞 240자를 출력합니다. 예시(축약):

```text
{'relic_id': 'nmm-bon-002789-00', 'chunk_id': '...', ...} 크기: 높이 81.5cm, 불신높이 50cm
```

`--relic-label "본관 2789"`를 붙이면 이번 질문만 YAML의 대상을 덮어씁니다. 순서와 개수는 Retriever 설정과 검색 결과에 따릅니다.

### 5-2. Hugging Face 모델을 양자화하여 답변 생성

exp01 YAML의 llm을 아래처럼 설정합니다. 다른 실험 설정은 유지합니다.

```yaml
llm:
  enabled: true
  provider: transformers
  source: huggingface
  model: Qwen/Qwen3-1.7B
  quantization: nf4
  device: cuda:0
  compute_dtype: auto
  trust_remote_code: false
  max_input_tokens: 4096
  max_new_tokens: 512
  enable_thinking: false
  visitor_type: general
  temperature: 0.0
  save_dir: "ai/llm/model/qwen3-1.7b-{quantization}"
```

```powershell
uv run --extra hf python ai/rag/scripts/test_retriever.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --query "이 불상의 전체 높이는 얼마인가요?" --ask
```

`질문 임베딩 → 선택 유물 청크 검색 → 근거·질문으로 프롬프트 작성 → 모델 로딩 및 양자화 → 답변 생성` 순서입니다. 처음에는 모델을 다운로드할 수 있습니다. 원본 다운로드 캐시와 양자화 모델 저장 폴더는 별개입니다. **추론 명령은 양자화 모델을 save_dir에 자동 저장하지 않습니다.**

콘솔 출력은 JSON 파일이 아닌 Python 딕셔너리 표현입니다. 다음은 형식 예시이며 실제 생성 결과가 아닙니다.

```text
{'answer': '전체 높이는 81.5cm입니다. [nmm-bon-002789-00]', 'visitor_type': 'general', 'relic_id': 'nmm-bon-002789-00', 'retrieved_ids': ['nmm-bon-002789-00']}
```

같은 유물의 여러 청크가 검색되면 retrieved_ids에 같은 ID가 반복될 수 있습니다. 답변은 자동 파일 저장되지 않습니다.

### 5-3. 양자화 모델 저장

YAML의 source: huggingface를 유지하고 실행합니다.

```powershell
uv run --extra hf python ai/rag/scripts/prepare_llm.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml
```

위 설정이면 `ai/llm/model/qwen3-1.7b-nf4/`에 모델, 토크나이저, preparation.json을 저장합니다. 설정과 가중치 파일도 포함되며 세부 구성은 모델·라이브러리에 따라 다릅니다.

이번 실행만 방식과 경로를 지정할 때:

```powershell
uv run --extra hf python ai/rag/scripts/prepare_llm.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --quantization int8 --output ai/llm/model/qwen3-1.7b-int8
```

--quantization은 nf4, int4, int8, fp8, bf16, fp16, none을 받습니다. CLI는 4bit·8bit 별칭을 받지 않습니다. 옵션은 YAML 파일 자체를 수정하지 않습니다. 출력 폴더가 이미 있으면 덮어쓰지 않고 중단하므로 새 경로를 사용합니다.

콘솔 출력 예시:

```text
모델과 토크나이저 저장 완료: .../ai/llm/model/qwen3-1.7b-nf4
재사용 설정: llm.source: local, llm.model: ".../ai/llm/model/qwen3-1.7b-nf4"
```

### 5-4. 저장한 모델로 답변 생성

YAML의 source와 model을 변경합니다. enabled: true, provider: transformers 및 생성 설정은 유지합니다.

```yaml
llm:
  enabled: true
  provider: transformers
  source: local
  model: ai/llm/model/qwen3-1.7b-nf4
```

```powershell
uv run --extra hf python ai/rag/scripts/test_retriever.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --query "이 불상의 전체 높이는 얼마인가요?" --ask
```

저장된 양자화 설정과 모델을 읽어 CUDA에 올립니다. LLM 로딩은 local_files_only=True를 사용합니다. 임베딩은 별도이므로 캐시가 없으면 다운로드할 수 있습니다. 출력은 5-2와 같으며 Ollama에 등록할 필요는 없습니다.

### 5-5. Ollama 모델로 답변 생성

이미 사용하는 Ollama 서버와 모델이 있다면 다음처럼 설정합니다. model에는 ollama list에 실제 존재하는 태그를 넣습니다.

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen3:8b-q4_K_M
  visitor_type: general
  temperature: 0.0
```

```powershell
ollama list
uv run python ai/rag/scripts/test_retriever.py --config ai/llm/configs/experiments/exp01_recursive_500.yaml --query "이 불상의 전체 높이는 얼마인가요?" --ask
```

Ollama 서버가 실행 중이어야 합니다. 현재 코드는 별도 YAML 서버 주소 옵션 없이 ChatOllama 기본 연결을 사용합니다. 모델이 없다면 사용 가능한 태그로 바꾸거나 먼저 준비합니다. llm.quantization은 Ollama 모델을 재양자화하지 않습니다. 답변 출력은 5-2와 같습니다.

## 실행 중 자주 확인할 사항

| 증상 | 확인 방법 |
|---|---|
| 기존 DB가 있다는 오류 | 기존 DB 평가는 evaluate_existing.py, 교체 구축은 run_experiment.py --rebuild |
| DB가 없다는 오류 | experiment 이름·base_dir·구축 여부 확인. 샘플 DB인지도 확인 |
| Gold 범위에 relic_label이 없다는 오류 | 내부 공백을 포함한 실제 번호와 Gold 포함 여부 확인 |
| 평가 N번 오류 | 해당 JSON 항목의 질문·번호·정답 배열 확인 |
| Recall이 계속 0 | 임시 정답 문자열인지, 현재 실험의 실제 청크 ID인지, 대상 유물이 맞는지 확인 |
| LLM 비활성화 오류 | --ask 사용 시 llm.enabled: true 지정 |
| CUDA를 사용할 수 없다는 오류 | CUDA 지원 PyTorch·드라이버·GPU 환경 확인. CPU 대체 실행은 없음 |
| 입력 토큰 제한 오류 | 검색 k·청크 크기를 줄이거나 모델 한도 내에서 입력 제한 조정 |
| 모델 저장 경로가 이미 있음 | prepare_llm.py --output으로 새 폴더 지정 |

구현 검증 테스트는 다음처럼 실행합니다. 실제 정답셋으로 검색 성능을 평가하는 것과는 별개입니다.

```powershell
uv run python -m unittest discover -s ai/rag/tests -v
```

성공하면 테스트 개수와 OK가 출력됩니다. 모델 관련 테스트는 모의 객체를 사용하므로 통과했다고 모든 양자화 방식의 실제 GPU 실행·저장이 검증되는 것은 아닙니다.
