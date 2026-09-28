# API 명세서 (국립중앙박물관 AI 도슨트)

`docs/feature_spec.md`의 기능을 기준으로 한 FastAPI 엔드포인트 명세.
이 문서에 정의된 요청/응답 스키마는 팀원 간 합의 없이 임의로 변경하지 않는다 (`Git_규칙.md` 13번).

## 0. 공통 사항

- Base URL: `/api/v1`
- Content-Type: 이미지 업로드는 `multipart/form-data`, 그 외는 `application/json`
- 공통 에러 응답:

```json
{
  "error": {
    "code": "ARTIFACT_NOT_FOUND",
    "message": "artifact_id에 해당하는 유물을 찾을 수 없습니다."
  }
}
```

- `visitor_type` 값은 반드시 다음 중 하나: `"child" | "general" | "expert"`

---

## 1. 유물 인식

### `POST /api/v1/classify`

업로드된 이미지를 EfficientNet 모델로 분류한다.

**Request** (`multipart/form-data`)

| 필드 | 타입 | 설명 |
|---|---|---|
| `image` | file | 유물 사진 (jpg/png) |

현재 구현 제한:

- 허용 형식: JPG/PNG
- 기본 최대 파일 크기: 10MB
- 기본 최대 해상도: 2,500만 픽셀
- 기본 confidence 임계값: 0.60 (실제 촬영 validation 데이터 확보 후 조정)
- 위 제한과 임계값은 서버 환경변수로 설정 가능

**Response 200**

```json
{
  "artifact_id": "bon002789",
  "artifact_name": "금동 반가사유상",
  "confidence": 0.94,
  "candidates": [
    { "artifact_id": "bon002789", "confidence": 0.94 },
    { "artifact_id": "jub002084", "confidence": 0.03 }
  ]
}
```

**Response 200 (신뢰도 낮음 / 인식 실패)**

```json
{
  "artifact_id": null,
  "artifact_name": null,
  "confidence": 0.0,
  "candidates": [],
  "message": "유물을 인식하지 못했습니다. 다시 촬영해주세요."
}
```

담당: Vision (`ai/vision/`) — 추론 로직, Backend — 라우팅/응답 조립

---

## 2. 유물 설명 조회

### `POST /api/v1/docent/description`

인식된 유물에 대해 관람객 유형에 맞는 설명을 반환한다.

**Request**

```json
{
  "artifact_id": "bon002789",
  "visitor_type": "general"
}
```

**Response 200**

```json
{
  "artifact_id": "bon002789",
  "artifact_name": "금동 반가사유상",
  "visitor_type": "general",
  "description": "이 불상은 한쪽 다리를 다른 쪽 무릎 위에 올리고 깊은 생각에 잠긴 모습을 표현한...",
  "sources": ["국립중앙박물관"]
}
```

**Response 404** (artifact_id 없음)

```json
{
  "error": {
    "code": "ARTIFACT_NOT_FOUND",
    "message": "artifact_id 'xxx'에 해당하는 유물을 찾을 수 없습니다."
  }
}
```

담당: RAG (`ai/rag/`) — 검색/컨텍스트 구성, Backend — DB 조회 및 응답 조립

---

## 3. 추가 질문 (채팅)

### `POST /api/v1/docent/chat`

인식된 유물에 대해 관람객이 자유 질문을 하고, RAG + Local LLM이 답변한다.

**Request**

```json
{
  "artifact_id": "bon002789",
  "visitor_type": "general",
  "session_id": "sess_20260923_abc123",
  "question": "이 불상은 왜 반가사유상이라고 불러요?"
}
```

- `session_id`가 없으면 서버가 새로 발급하여 응답에 포함한다 (최초 질문 시).

**Response 200**

```json
{
  "session_id": "sess_20260923_abc123",
  "artifact_id": "bon002789",
  "visitor_type": "general",
  "answer": "반가사유상이라는 이름은 '반가부좌'를 하고 '사유'(생각)에 잠긴 모습에서 유래했습니다...",
  "sources": ["국립중앙박물관"]
}
```

담당: RAG (`ai/rag/`) — 관련 문서 검색, Local LLM (`ai/llm/`) — 답변 생성, Backend — 세션/대화 이력 관리(SQLite)

---

## 4. 유물 메타데이터 조회

### `GET /api/v1/artifacts/{artifact_id}`

유물 기본 정보를 조회한다 (관리/디버깅용).

**Response 200**

```json
{
  "artifact_id": "bon002789",
  "accession_no": "본관 2789",
  "artifact_name": "금동 반가사유상",
  "source": "국립중앙박물관",
  "description": "...",
  "designation_no": "국보 제1962-1호"
}
```

`accession_no`는 국립중앙박물관 소장품 관리번호(예: 본관 2789, 신수 1794), `designation_no`는 국보/보물 등 문화재 지정번호로 서로 다른 값이다. 두 필드 모두 없을 수 있으며 그 경우 `null`을 반환한다.

**Response 404**: 위 공통 에러 형식과 동일

담당: Backend

---

## 5. 대화 이력 조회 (선택 구현)

### `GET /api/v1/docent/chat/{session_id}`

세션의 전체 대화 이력을 조회한다.

**Response 200**

```json
{
  "session_id": "sess_20260923_abc123",
  "artifact_id": "bon002789",
  "visitor_type": "general",
  "messages": [
    { "role": "user", "content": "이 불상은 왜 반가사유상이라고 불러요?" },
    { "role": "assistant", "content": "반가사유상이라는 이름은..." }
  ]
}
```

담당: Backend

---

## 6. 엔드포인트 요약

| Method | Path | 설명 | 주 담당 |
|---|---|---|---|
| POST | `/api/v1/classify` | 이미지 → 유물 분류 | Vision, Backend |
| POST | `/api/v1/docent/description` | 유물 설명 조회 | RAG, Backend |
| POST | `/api/v1/docent/chat` | 추가 질문 응답 | RAG, LLM, Backend |
| GET | `/api/v1/artifacts/{artifact_id}` | 유물 메타데이터 조회 | Backend |
| GET | `/api/v1/docent/chat/{session_id}` | 대화 이력 조회 | Backend |

## 7. 변경 이력

- 최초 작성 (draft) — 실제 구현 중 필드가 추가/변경되면 이 표와 각 섹션을 함께 수정하고, PR에 변경 사유를 남긴다.
- `/api/v1/classify` 구현 — 이미지 검증, confidence 임계값, top-3 후보 및 공통 오류 형식 반영
- `data/metadata.csv`에 `accession_no`(소장품 관리번호) 컬럼 추가, `/api/v1/artifacts/{artifact_id}` 응답에 반영
