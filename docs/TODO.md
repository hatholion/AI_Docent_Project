# TODO

## metadata — 완료

7개 유물의 `artifact_id`, `accession_no`, `artifact_name`, `description`이 모두 확정됨
(`data/metadata.csv` 참고). `description`은 `ai/rag/build_gold.py`로 e뮤지엄 SILVER
데이터(`data/rag/silver/emuseum/`)에서 `accession_no` 기준으로 매칭해 공식 설명문을
그대로 가져왔다.

국보/보물 지정 현황 (SILVER `designations`/`designation_nos` 기준):

| artifact_id | 유물명 | 지정 여부 |
|---|---|---|
| `bon002789` | 금동 반가사유상 | 국보 제1962-1호 |
| `jub000702` | 백자 달항아리 | 보물 1437호 |
| `ssu001794` | 농경문 청동기 | 보물 1823호 |
| `duk000798` | 청동촛대 | 지정 없음 |
| `ssu001846` | 방패형 동기 | 지정 없음 |
| `ssu003094` | 요령식 동검 | 지정 없음 |
| `ssu022891` | 빗살무늬토기 | 지정 없음 |

## 남은 작업

- [ ] `jub000702`(백자 달항아리) — `data/processed/{train,val,test}/jub000702/` 아직 비어있음.
      `scripts/split_dataset.py` 다시 실행해서 채워야 함
- [ ] `jub000702` 데이터로 Vision 모델 재학습 필요 (기존 checkpoint는 `jub002084` 기준으로
      학습된 것이라 클래스 구성이 달라짐 — `Tomorrowdo.md` 참고)
- [ ] 실제 핸드폰 촬영 데이터 수집 (`Tomorrowdo.md` "이어서 할 작업" 참고)
