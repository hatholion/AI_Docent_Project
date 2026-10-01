# LLM Judge 결과 (3개 모델이 돌아가며 심판, temperature=0)

유효 채점: 40 / 42 (JSON 파싱 실패 제외)

| 모델 | correctness | relevance | groundedness | fluency | 평균 | 승수(무승부 제외) |
|---|---:|---:|---:|---:|---:|---:|
| exaone3.5:7.8b | 4.65 | 4.80 | 4.72 | 4.55 | 4.68 | 6 |
| llama3.1:latest | 4.40 | 4.72 | 4.53 | 4.42 | 4.52 | 5 |
| qwen2.5-7b-quant-manual | 4.65 | 4.78 | 4.75 | 4.47 | 4.66 | 6 |

무승부(동점): 23건

## ⚠️ 중요: self-preference bias 실측 사례

`ssu003094`(요령식 동검) 지정문화재 질문에서 Llama 3.1이 존재하지 않는
"보물 제1239-2번"을 지어낸 사례(`ai/llm/comparison_summary.md` 참고)를
심판별로 보면:

| 심판 | groundedness 점수 |
|---|---:|
| exaone3.5:7.8b (제3자) | 1 |
| **llama3.1:latest (자기 자신)** | **5 (만점)** |
| qwen2.5-7b-quant-manual (제3자) | 5 |

**Llama가 자기 답변을 채점할 때 자신의 hallucination을 전혀 못 잡아냈다.**
위 표의 groundedness 평균(4.53)은 이 편향이 섞여서 실제보다 후하게 나온
수치다. 그래서 이 LLM Judge 점수만으로 모델을 판단하지 말고, 반드시
`ai/llm/comparison_summary.md`의 자동 사실 검증 결과, `ai/llm/human_eval_blind.md`의
사람 평가와 같이 봐야 한다.
