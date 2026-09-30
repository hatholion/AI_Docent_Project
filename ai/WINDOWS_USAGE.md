# Windows 사용 가이드

이 문서는 Windows PowerShell에서 박물관 AI 도슨트를 설치하고 실행하는 방법을 설명한다.
이 프로젝트에서는 Windows와 WSL의 가상환경을 함께 사용하지 않는다.

## 1. 준비 사항

다음 프로그램이 설치되어 있어야 한다.

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- NVIDIA GPU 사용 시 최신 그래픽 드라이버

Windows PowerShell에서 프로젝트 폴더로 이동한다.

```powershell
cd C:\Users\Admin\llm_project
```

이후의 모든 명령은 프로젝트 루트에서 실행한다.

## 2. 캐시 경로 설정

Git으로 clone한 직후에는 `.hf-cache`와 `.uv-cache` 폴더가 없는 것이 정상이다. 아래
환경변수는 기존 폴더를 찾는 설정이 아니라 캐시를 생성할 위치를 지정하는 설정이다.
폴더를 직접 만들 필요는 없다.

새 PowerShell 창을 열 때 다음 환경변수를 한 번 설정한다. 같은 창에서는 다시 설정할
필요가 없다.

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
```

`HF_HOME`에는 `BAAI/bge-m3` 같은 Hugging Face 임베딩 모델이 저장된다. EXAONE은
Hugging Face 캐시가 아니라 Ollama의 모델 저장소를 사용한다.

- `.uv-cache`: `uv sync`를 처음 실행할 때 자동 생성된다.
- `.hf-cache`: 임베딩 모델을 처음 내려받을 때 자동 생성된다.
- `.venv`: `uv sync`가 Windows용 Python 가상환경으로 자동 생성한다.

이 폴더들은 로컬 실행 중 생성되는 파일이므로 Git으로 공유하지 않는다. 새로 clone한
사람은 최초 `uv sync`와 최초 모델 실행 때 각각 필요한 파일을 내려받게 된다. 따라서
첫 설치와 첫 추론에는 인터넷 연결이 필요하다.

## 3. Python 환경 설치

```powershell
uv sync
```

처음 실행하면 의존성을 설치하므로 시간이 걸릴 수 있다. 이후에는 변경된 패키지만
반영한다.

> Windows PowerShell과 WSL에서 같은 `.venv`를 번갈아 사용하지 않는다. WSL에서 만든
> 가상환경은 Windows에서 사용할 수 없으며, 반대의 경우도 마찬가지다.

## 4. Ollama와 EXAONE 확인

Ollama가 실행 중인지 확인하고 설치된 모델을 조회한다.

```powershell
ollama list
```

현재 `ai/llm/configs/base.yaml`의 모델이 `exaone3.5:7.8b`라면 목록에 같은 태그가
있어야 한다. 없다면 한 번 내려받는다. (모델은 yaml파일에서 변경 가능)

```powershell
ollama pull exaone3.5:7.8b
```

현재 메모리에 올라간 모델과 CPU/GPU 사용 여부는 다음 명령으로 확인한다.

```powershell
ollama ps
```

## 5. 직접 입력하는 멀티턴 대화

다음 명령은 `본관 2789`를 기준 유물로 선택하고 대화형 모드를 시작한다. (초기 유물은 변경가능 - vision모델에서 전달받은 모델로 변경)

```powershell
uv run python ai/llm/chat.py `
  --config ai/llm/configs/base.yaml `
  --relic-label "본관 2789" `
  --interactive
```

최초 유물 설명이 출력된 뒤 `질문>`에 직접 입력한다.

```text
질문> 이 유물은 어떻게 만들어졌어?

질문> 비슷한 유물을 두 개 알려줘

질문> 그중 첫 번째 유물과 차이점은 뭐야?
```

같은 프로세스에서 `session_id`와 대화 이력이 유지되므로 앞선 질문과 답변을 참고할
수 있다. 다음 중 하나를 입력하면 종료한다.

```text
/exit
exit
quit
종료
```

`--relic-label`은 유물명이 아니라 대화의 기준 유물을 선택하는 소장품번호다. 후속
검색은 해당 번호 하나로 제한되지 않고 전체 Chroma DB를 대상으로 실행된다. 옵션을
생략하면 `base.yaml`의 `runtime.initial_relic_label`을 사용한다.

방문객 유형도 선택할 수 있다.

```powershell
uv run python ai/llm/chat.py `
  --config ai/llm/configs/base.yaml `
  --relic-label "본관 2789" `
  --visitor-type general `
  --interactive
```

선택 가능한 값은 다음과 같다.

- `child`: 어린이 대상의 쉬운 설명
- `general`: 일반 성인 대상의 도슨트 설명
- `expert`: 전문 용어와 미술사적 맥락을 포함한 설명

## 6. 한 번만 실행하기

기준 유물의 최초 설명만 출력한다.

```powershell
uv run python ai/llm/chat.py `
  --config ai/llm/configs/base.yaml `
  --relic-label "본관 2789"
```

최초 설명 뒤 후속 질문 하나까지 실행하고 최종 답변만 출력한다.

```powershell
uv run python ai/llm/chat.py `
  --config ai/llm/configs/base.yaml `
  --relic-label "본관 2789" `
  --question "이 유물과 비슷한 유물은?"
```

세션 ID, 검색 결과, 유사도 점수 등 전체 결과가 필요할 때만 `--json`을 추가한다.

```powershell
uv run python ai/llm/chat.py `
  --config ai/llm/configs/base.yaml `
  --relic-label "본관 2789" `
  --question "이 유물과 비슷한 유물은?" `
  --json
```

## 7. 백엔드 서버 실행

실제 서비스에서는 요청마다 CLI를 새로 실행하지 않고 백엔드를 계속 실행한다. 첫 번째
PowerShell에서 캐시 환경변수를 설정한 뒤 서버를 시작한다.

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

두 번째 PowerShell에서 최초 설명을 요청한다.

```powershell
$initial = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/docent/description" `
  -ContentType "application/json" `
  -Body (@{
    relic_label = "본관 2789"
    visitor_type = "general"
  } | ConvertTo-Json)

$sessionId = $initial.session_id
$initial.answer
```

같은 `$sessionId`로 후속 질문을 보낸다.

```powershell
$reply = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/docent/chat" `
  -ContentType "application/json" `
  -Body (@{
    session_id = $sessionId
    question = "이 유물은 어떻게 만들어졌어?"
  } | ConvertTo-Json)

$reply.answer
```

후속 요청마다 같은 `$sessionId`를 보내야 멀티턴 대화가 이어진다. API 응답에는 내부
처리를 위한 검색 결과와 메타데이터도 포함되지만 사용자 화면에는 `answer`만 표시한다.
현재 세션은 프로세스 메모리에 저장되므로 백엔드를 재시작하면 사라진다.

## 8. 자동 멀티턴 검사

다음 명령은 정해진 네 가지 시나리오로 세션 이력, 질의 재작성, 전체 DB 검색 연결을
검사한다.

```powershell
uv run python ai/rag/scripts/smoke_multiturn.py
```

이 스크립트는 대화형 챗봇이 아니므로 결과 JSON을 출력한 뒤 정상 종료한다. 실제
EXAONE과 직접 대화하려면 `--interactive` 모드를 사용한다.

전체 단위 테스트는 다음 명령으로 실행한다.

```powershell
uv run python -m unittest discover -s ai/rag/tests -v
```

## 9. 자주 발생하는 문제

### `Batches: 100%`가 매 질문마다 표시됨

새 질문을 검색 벡터로 변환하는 진행 표시다. 임베딩 모델을 매번 다시 내려받거나
로딩하는 것이 아니다. 전체 DB를 검색하려면 질문 임베딩 계산은 매 턴 필요하다.

### 첫 답변이 오래 걸림

최초 실행에서는 임베딩 모델, Chroma DB, Ollama 모델을 메모리에 올려야 한다. 같은
대화형 프로세스나 백엔드 서버의 후속 요청은 초기 로딩을 재사용한다. `exaone3.5:7.8b`는
작은 모델보다 생성 시간이 길며, `ollama ps`에서 CPU/GPU 사용 상태를 확인할 수 있다.

### `.venv\lib64` 액세스 거부

WSL에서 생성한 `.venv`를 Windows의 `uv`가 교체하려 할 때 발생할 수 있다. 실행 중인
WSL 터미널을 닫고 필요한 경우 다음 명령으로 WSL을 종료한 뒤, WSL용 `.venv`의 이름을
변경하고 Windows PowerShell에서 `uv sync`를 다시 실행한다.

```powershell
wsl --shutdown
Rename-Item -LiteralPath ".\.venv" -NewName ".venv-wsl-backup"
uv sync
```

백업 이름이 이미 존재한다면 사용하지 않은 다른 이름을 지정한다.

### 모델을 찾을 수 없음

`base.yaml`의 `llm.model` 값과 `ollama list`의 모델 태그가 정확히 같은지 확인한다.

```powershell
ollama list
Get-Content ai\llm\configs\base.yaml
```

### 세션을 찾을 수 없음

백엔드를 재시작했거나 다른 서버 프로세스로 요청한 경우다. `/description`을 다시
호출해 새 `session_id`를 받고, 이후 모든 `/chat` 요청에 그 값을 전달한다.
