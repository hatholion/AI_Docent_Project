# AI 도슨트 API 명세

현재 구현 기준 API다. 서버 실행:

```powershell
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 최초 유물 설명

`POST /api/v1/docent/description`

```json
{
  "relic_label": "본관 2789",
  "session_id": null,
  "visitor_type": "general"
}
```

`relic_label`을 생략하면 YAML `runtime.initial_relic_label`을 사용한다. 전달하면 YAML보다 우선한다. Gold exact lookup만 사용하며 Retriever는 호출하지 않는다. 응답은 `session_id`, `current_relic_label`, `answer`, `source_chunk_ids`, `retrieval_used=false`를 포함한다.

`visitor_type`은 `child`, `general`, `expert` 중 하나다. 생략하면 YAML `llm.visitor_type`을 사용하며, 선택한 유형의 안내문이 답변 생성용 system prompt에 추가된다.

## 후속 질문

`POST /api/v1/docent/chat`

```json
{
  "session_id": "sess_...",
  "relic_label": null,
  "visitor_type": "general",
  "question": "이 유물과 같은 재질의 다른 유물은?"
}
```

기존 session의 `current_relic_label`과 최근 history로 query를 재작성한 뒤 전체 Chroma collection을 검색한다. 요청에 `relic_label`을 보내면 해당 session의 현재 유물을 변경한다. `session_id` 없이 호출하면 요청 label 또는 YAML 기본 label로 새 session을 만든다.

응답에는 `retrieval_query`, `answer`, 검색된 각 `chunk_id`, `relic_label`, `score`, `distance`가 포함된다.

## 대화 이력

`GET /api/v1/docent/chat/{session_id}`

```json
{
  "session_id": "sess_...",
  "current_relic_label": "본관 2789",
  "history": [
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

현재 session store는 in-memory이므로 서버 재시작 시 사라진다.

## 오류

- 404: `relic_label` 또는 `session_id`를 찾을 수 없음
- 422: 빈 질문, 잘못된 visitor type 등 입력/설정 오류
- 503: Chroma 미구축, embedding/LLM 서버·모델 문제 등 런타임 오류
