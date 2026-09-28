# 기능 명세서 (국립중앙박물관 AI 도슨트)

## 1. 목표

관람객이 유물 사진을 업로드하면 AI가 유물을 인식하고, 관람객 유형(어린이/일반인/전문가)에 맞춘 설명을 제공하며,
추가로 궁금한 점을 자유롭게 물어볼 수 있는 도슨트 서비스를 만든다.

## 2. 핵심 사용자 흐름

```text
1) 관람객 유형 선택 (어린이 / 일반인 / 전문가)
        ↓
2) 유물 사진 업로드 (촬영 또는 파일 업로드)
        ↓
3) Vision 모델이 유물 분류 → artifact_id 반환
        ↓
4) artifact_id + visitor_type 기준으로 눈높이에 맞는 설명 제공
        ↓
5) (선택) 관람객이 추가 질문 입력 → RAG + Local LLM이 답변
        ↓
6) 같은 유물에 대해 대화를 이어가거나, 새 유물 인식으로 돌아감
```

- 화면 흐름은 `docs/frontend_plan.md`, 디자인 시안은 `docs/design/` 참고.
- 관람객 유형 값은 프론트~백엔드~LLM 프롬프트까지 `visitor_type: "child" | "general" | "expert"`로 통일한다.

## 3. 기능 모듈 및 담당 영역

`Git_규칙.md`의 역할 분담(9~10번)을 기준으로 한다.

| 모듈 | 담당 디렉터리 | 주요 기능 |
|---|---|---|
| Vision (분류) | `ai/vision/` | EfficientNet 파인튜닝, 이미지 → `artifact_id` + `confidence` 추론 |
| RAG (검색) | `ai/rag/` | `artifact_id` 기준 공식 설명/문서 검색, LLM에 넘길 context 구성 |
| Local LLM | `ai/llm/` | RAG context + `visitor_type` + 사용자 질문 → 답변 생성 (로컬 추론) |
| Backend | `backend/` | FastAPI 라우팅, DB(SQLite) 연동, 각 모듈 오케스트레이션 |
| Frontend | `frontend/` | React UI (유형 선택 / 업로드 / 결과·채팅 화면) |
| 공통 데이터 | `data/` | 학습 이미지, 3D 원본, `metadata.csv` |

## 4. 기능 상세

### 4.1 유물 인식 (Vision)

- 입력: 업로드된 유물 이미지 1장
- 처리: `data/processed/train|val|test/<artifact_id>/` 구조로 파인튜닝된 EfficientNet 모델로 추론
- 출력: `artifact_id`, `confidence` (필요 시 상위 N개 후보)
- 신뢰도가 낮을 경우(예: threshold 미달) 재촬영 유도 등 예외 처리 필요 (임계값은 Vision 담당자가 결정 후 문서화)

### 4.2 유물 설명 제공 (RAG + Backend)

- 입력: `artifact_id`, `visitor_type`
- 처리: `data/metadata.csv` / RAG 검색 결과를 바탕으로 `visitor_type`별 난이도에 맞는 설명 생성
  - 어린이: 쉬운 어휘, 짧은 문장
  - 일반인: 친절하고 이해하기 쉬운 설명
  - 전문가: 학술적 용어, 상세 정보(시대, 재질, 지정번호 등) 포함
- 출력: 설명 텍스트 + 출처

### 4.3 추가 질문 (RAG + Local LLM)

- 입력: `artifact_id`, `visitor_type`, 사용자 질문, (선택) 이전 대화 맥락
- 처리: RAG로 관련 문서 검색 → Local LLM이 `visitor_type`에 맞춰 답변 생성
- 출력: 답변 텍스트 + 참고 출처
- 대화 이력은 세션 단위로 SQLite에 저장 (동일 유물에 대해 이어지는 대화 지원)

### 4.4 DB (SQLite, 초안)

```text
artifacts
├─ artifact_id     TEXT PRIMARY KEY   -- 예: bon002789 (내부 관리용 ID, 폴더명/API 값과 동일)
├─ accession_no    TEXT               -- 소장품 관리번호 (예: 본관 2789)
├─ artifact_name   TEXT               -- 예: 금동 반가사유상
├─ source          TEXT               -- 예: 국립중앙박물관
├─ description     TEXT               -- 공식 설명 원문 (RAG 소스)
└─ designation_no  TEXT NULL          -- 국보/보물 지정번호 (예: 국보 제1962-1호), accession_no와 별개

conversations
├─ session_id      TEXT PRIMARY KEY
├─ artifact_id     TEXT (FK -> artifacts.artifact_id)
├─ visitor_type    TEXT               -- child | general | expert
├─ created_at      DATETIME

messages
├─ id              INTEGER PRIMARY KEY AUTOINCREMENT
├─ session_id      TEXT (FK -> conversations.session_id)
├─ role            TEXT               -- user | assistant
├─ content         TEXT
└─ created_at      DATETIME
```

- `artifacts` 테이블은 `data/metadata.csv`를 초기 시드 데이터로 사용한다 (컬럼 대응: `artifact_id`, `accession_no`, `artifact_name`, `source`, `description`).
- 정확한 스키마/마이그레이션 방식은 Backend 담당자가 확정 후 `docs/`에 반영한다.

## 5. 비기능 요구사항

- Local LLM 사용으로 인한 추론 지연을 고려해 API는 비동기 처리 또는 로딩 상태 표시를 전제로 설계한다.
- 인터넷 연결 없이도 핵심 기능(인식~설명)이 동작하는 것을 목표로 한다 (모델/DB 모두 로컬 구동).
- 공용 인터페이스(요청/응답 스키마)는 `docs/api_spec.md` 기준으로 하며, 임의 변경 금지 (`Git_규칙.md` 13번).

## 6. 미정 사항 (TODO)

- Vision 분류 신뢰도 임계값
- RAG에 사용할 벡터DB/임베딩 모델 선택
- Local LLM 모델 선택 및 양자화 방식
- `visitor_type`별 프롬프트 템플릿
- 인증/세션 관리 방식 (관람객을 어떻게 식별할지, 익명 세션인지 등)
