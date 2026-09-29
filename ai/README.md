# Gold 기반 멀티턴 AI 도슨트

이 문서는 현재 실행되는 `Gold → Embedding → Chroma → Retriever → Multi-turn LLM` 구조를 설명한다. 모든 명령은 프로젝트 루트에서 실행한다.

## 데이터 원칙과 실제 스키마

입력은 이미 완성된 `data/gold/gold0929.jsonl`이다. 런타임은 Raw/Silver를 읽거나 Gold를 재생성·재선정·재청킹하지 않는다. Gold 한 줄은 기존 chunk 하나이며, 그 한 줄이 embedding document 및 Chroma document 하나가 된다. 기존 `chunk_id`가 Chroma ID로 그대로 쓰인다.

현재 검증된 값은 다음과 같다.

- Gold record 및 고유 `chunk_id`: 4,269개
- 고유 `parent_id`: 3,651개
- 고유한 비어 있지 않은 `relic_label`: 3,551개
- `relic_label`이 null인 set record: 100개
- 비어 있는 `text`: 0개

실제 최상위 필드는 아래 8개다.

```text
chunk_id, chunk_type, parent_type, parent_id,
relic_label, text, char_len, metadata
```

`metadata`에는 다음 필드가 있다.

```text
record_type, collection_name, nationality, period,
material_l1, material_l2, purpose_l1, purpose_l2,
find_place_l1, find_place_l2, designations,
has_description, has_image
```

일부 유물은 `profile`과 `description` record를 각각 가지므로 같은 `relic_label`이 여러 record에 존재한다. 최초 설명은 하나를 임의 선택하지 않고 해당 label의 기존 Gold record를 모두 사용한다. `relic_label=null`인 set record도 DB에서 제외하지 않으며 `parent_id`로 식별한다. 단, label 직접 조회 대상이 될 수는 없다.

`chunk_id`는 의미 검색 조건이 아니다. Chroma 문서 식별, Gold 원문 연결, 검색 결과 확인, 평가 정답 비교, 디버깅에 사용한다. 검색 순위는 query embedding과 document embedding 사이의 cosine similarity로 결정된다.

## Embedding document와 Chroma metadata

`ai/rag/document_builder.py`는 Python dict 표현 대신 실제 Gold 필드에 한국어 의미 라벨을 붙인다.

```text
소장품번호: ...
Gold 레코드 유형: ...
국적: ...
시대: ...
재질 대분류: ...
재질 세부: ...
용도 대분류: ...
용도 세부: ...
출토지 광역: ...
출토지 상세: ...
지정 유형: ...

유물 정보:
<Gold text 원문>
```

값이 없는 필드는 생략한다. 내부 식별자는 검색 본문에 억지로 넣지 않지만 Chroma metadata에는 아래처럼 유지한다.

```text
chunk_id, chunk_type, parent_type, parent_id, relic_label,
collection_name, record_type, nationality, period,
material_l1, material_l2, purpose_l1, purpose_l2,
find_place_l1, find_place_l2, designations
```

Chroma가 scalar metadata만 지원하므로 `designations` 배열은 쉼표로 연결한다. null `relic_label`은 metadata에서 빈 문자열이며 Retriever 반환 시 다시 `null`로 정규화한다.

## YAML 설정

기본 설정은 `ai/llm/configs/base.yaml` 하나다.

| 설정 | 의미 |
|---|---|
| `data.gold_path` | 완성된 Gold JSONL |
| `runtime.initial_relic_label` | CLI/demo 최초 설명의 기본 label |
| `embedding.model` | 문서와 query가 함께 사용하는 embedding model |
| `embedding.device` | `auto`, `cpu`, `cuda` 등 |
| `embedding.batch_size` | embedding batch 크기 |
| `embedding.normalize_embeddings` | 벡터 정규화 여부 |
| `embedding.query_prefix` / `document_prefix` | 모델별 선택 접두어 |
| `vector_db.provider` | 현재 `chroma`만 지원 |
| `vector_db.persist_directory` | 영속 DB 폴더 |
| `vector_db.collection_name` | Chroma collection 이름 |
| `vector_db.add_batch_size` | Chroma upsert batch 크기 |
| `retrieval.top_k` | 기본 검색 결과 수 |
| `chat.history_turns` | Query Rewrite와 답변 prompt에 넣는 최근 turn 수 |
| `query_rewrite.enabled` | 후속 질문 standalone query 재작성 여부 |
| `llm.enabled` | LLM 호출 허용 여부 |
| `llm.provider` | `ollama` 또는 `transformers` |
| `llm.model` | Ollama tag, Hugging Face ID 또는 로컬 모델 경로 |
| `llm.visitor_types` | 사용자 선택지 `child`, `general`, `expert` |
| `llm.visitor_type` | API/CLI에서 생략했을 때 사용할 기본 방문객 유형 |
| `llm.temperature` | 생성 temperature |
| `llm.top_p` | Ollama nucleus sampling 값 |
| `llm.num_predict` | Ollama 최대 생성 token |
| `llm.num_ctx` | Ollama context 크기 |
| `llm.reasoning` | Ollama reasoning 출력을 사용할지 여부. 현재 Qwen3은 답변 token 확보를 위해 `false` |

Transformers 전용 `source`, `quantization`, `device`, `compute_dtype`, `trust_remote_code`, `max_input_tokens`, `max_new_tokens`, `enable_thinking`은 기존 직접 로딩 구현에서 계속 사용한다.

우선순위는 함수/API argument > CLI argument > YAML > 코드 기본값이다. 예를 들어 API/CLI의 `relic_label`이 있으면 `runtime.initial_relic_label`보다 우선한다. 평가의 `--top-k`도 YAML을 덮어쓴다.

설정을 읽을 때 Gold 경로, embedding model, 프로젝트 내부 Chroma 경로, collection 이름, 양수 `top_k`, 0 이상 `history_turns`, LLM provider/model 및 실제 generation option을 검증한다.

## 설치와 Vector DB 구축

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:HF_HOME = "$PWD/.hf-cache"
uv sync
uv run python ai/rag/scripts/build_vector_db.py --config ai/llm/configs/base.yaml --rebuild
```

기본 embedding model은 기존 프로젝트에서 사용하던 `BAAI/bge-m3`다. Gold document와 query 모두 `ai/rag/embeddings.py`의 같은 model instance를 사용한다. 모델이나 정규화·prefix를 바꾸면 반드시 `--rebuild`로 Chroma를 다시 구축해야 한다.

첫 실행은 Hugging Face에서 모델을 내려받아야 한다. 다운로드가 끝난 제한 네트워크 환경에서는 `$env:HF_HUB_OFFLINE = "1"`로 캐시만 사용할 수 있다.

기본 DB는 `ai/rag/storage/chroma`, collection은 `museum_relics`다. `--rebuild`는 지정 collection만 삭제하고 재생성한다. `--rebuild` 없이 다시 실행하면 같은 `chunk_id`로 upsert하므로 중복이 쌓이지 않는다. 완료 전에 Gold/Chroma 개수, 전체 ID 집합, 빈 문서, 필수 metadata를 검증한다.

## 최초 설명과 멀티턴 흐름

최초 설명은 Retriever를 호출하지 않는다.

```text
relic_label → Gold exact lookup → 같은 label의 Gold record → LLM → session 저장
```

label이 없으면 유사 유물을 선택하지 않고 404/`RelicNotFoundError`로 처리한다.

후속 질문은 다음 순서다.

```text
question + current_relic_label + recent history
→ Query Rewrite
→ 동일 embedding model
→ 전체 Chroma collection similarity search
→ current relic Gold + 검색 결과 + history + question
→ LLM
→ user/assistant message 저장
```

Query Rewrite prompt와 최종 답변 prompt는 분리되어 있다. 안정성을 위해 옵션이 켜진 모든 후속 질문을 짧은 standalone query로 재작성한다. Retriever에는 `relic_label` filter를 전달하지 않으므로 다른 유물·관련 유물·비교 질문을 검색할 수 있다. 검색 결과가 현재 유물 Gold와 같은 `chunk_id`이면 최종 prompt에서 중복만 제거하고, 원래 검색 결과 목록은 디버깅을 위해 보존한다.

세션은 `session_id`, `current_relic_label`, 전체 `history`를 가진다. 현재 구현은 thread-safe in-memory 저장소이므로 서버 재시작 시 세션이 사라진다. 전체 history는 저장하되 LLM에는 최근 `chat.history_turns`만 전달한다.

로컬 최초 설명 및 후속 질문:

```powershell
uv run python ai/llm/chat.py --config ai/llm/configs/base.yaml
uv run python ai/llm/chat.py --config ai/llm/configs/base.yaml --relic-label "신수 1794" --question "이 유물과 비슷한 유물은?"
```

Ollama 서버에 YAML의 model tag가 실제로 설치되어 있어야 한다.

## Retriever 결과와 평가

Retriever 결과는 Chroma raw tuple 대신 다음 공통 구조다.

```json
{
  "chunk_id": "...",
  "relic_label": "...",
  "content": "...",
  "metadata": {},
  "score": 0.0,
  "distance": 0.0
}
```

실제 Gold 기반 6종 smoke search:

```powershell
uv run python ai/rag/scripts/smoke_retrieval.py --config ai/llm/configs/base.yaml
```

실제 Gold ID만 사용한 30개 평가 표본은 `ai/llm/evaluation/evalset/example_retrieval_eval.json`에 있다. 기본 필드는 `id`, `question`, `relevant_chunk_ids`, `relevant_relic_labels`이고, `current_relic_label`과 `category`는 선택 항목이다. 사용자는 `relic_label`을 모른다고 가정하므로 `question`과 선택적 `retrieval_query`에는 소장품 번호형 `relic_label`을 넣지 않는다. `relevant_relic_labels`와 `current_relic_label`은 채점 및 시스템 상태용 메타데이터이므로 사용자 질의가 아니다. 평가 CLI는 Gold의 실제 `relic_label`이 질의에 포함되면 오류를 낸다. 대명사형 표본은 LLM 호출 없이 standalone Retriever 자체를 평가하도록 선택적 `retrieval_query`를 포함할 수 있다.

```powershell
uv run python ai/llm/evaluation/evaluate_retriever.py `
  --config ai/llm/configs/base.yaml `
  --dataset ai/llm/evaluation/evalset/example_retrieval_eval.json `
  --output ai/llm/evaluation/results/retrieval_eval.json
```

- Recall@K: Top-K에서 검색된 relevant chunk 수 / 전체 relevant chunk 수
- MRR: 첫 relevant chunk rank의 역수(없으면 0)를 query 전체에서 평균

상세 JSON에는 질문, 정답/검색 chunk ID와 label, score/distance, Recall@1/3/5, 첫 정답 rank가 저장된다.

## Backend

```powershell
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

- `GET /health`
- `POST /api/v1/docent/description`: `relic_label`, 선택 `session_id`, `visitor_type`
- `POST /api/v1/docent/chat`: `question`, 선택 `session_id`, `relic_label`, `visitor_type`
- `GET /api/v1/docent/chat/{session_id}`: 전체 세션 이력

`DOCENT_CONFIG` 환경변수로 다른 YAML 경로를 지정할 수 있다. 요청의 `relic_label`은 YAML 기본값보다 우선한다.

## 테스트

```powershell
uv run python -m unittest discover -s ai/rag/tests -v
uv run python -m compileall ai backend
```

단위 테스트는 실제 Gold 스키마/개수/기존 ID 보존, 1 record=1 Chroma document, persist/reload, 전체 collection 검색, 정규화 결과, 멀티턴 Query Rewrite와 history, Recall/MRR을 외부 LLM 없이 검사한다.

외부 LLM이 없는 환경에서 실제 Gold/Chroma까지 사용해 4개 멀티턴 시나리오를 점검하려면 다음을 실행한다.

```powershell
uv run python ai/rag/scripts/smoke_multiturn.py
```
