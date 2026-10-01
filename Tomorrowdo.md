# Vision 학습 현황 및 다음 작업

이 노트북 자체에 GPU(RTX 4060 Laptop)가 있어서, 별도 GPU 노트북으로 옮기는 절차 없이
이 문서에 있던 clone/환경 이전 계획은 더 이상 필요하지 않다. 환경 설치·학습·평가 명령은
전부 `README.md`의 "Vision 학습 환경", "분류 API 실행" 절에 정리되어 있으니 여기서는
중복 기재하지 않고, 현재 상태와 다음 작업만 남긴다.

## 발표/PPT용 핵심 수치 (v2, 2026-09-29 기준)

- **최종 checkpoint**: `runs/vision/fine_tune_gpu_v2/best_model.pth`
- **데이터**: synthetic 1,416장 + real(핸드폰 촬영) 66장, 7개 유물 클래스
  - train 945장 (전부 synthetic — real 사진은 train에 넣지 않음)
  - val 214장 (synthetic 202 + real 12)
  - test 257장 (synthetic 203 + real 54, **real 위주로 배치**)
- **test 전체 결과**: **accuracy 99.61%**, **macro F1 99.63%** (257장 중 1장만 오분류)
- **synthetic만**: 203/203 전부 정답 (100%)
- **real(실제 촬영)만**: 54장 중 53장 정답 — 유일한 오분류 건도 confidence **0.28**로
  낮게 나와, 실서비스 confidence 임계값(0.60) 적용 시 "재촬영 요청"으로 걸러짐
  (오답을 자신 있게 내놓은 게 아니라는 뜻)
- 이전 v1(synthetic만으로 검증)은 100%였지만 실제 사진 미검증 상태였음. v2는
  **real 사진으로 실측 검증까지 마친 수치**라는 점이 핵심 차별점

## 1. 완료된 상태

- [x] 7개 유물 클래스 메타데이터 등록 (`bon002789`, `duk000798`, `jub000702`, `ssu001794`, `ssu001846`, `ssu003094`, `ssu022891`)
- [x] synthetic + real 이미지 train/val/test 분할 (real은 val/test에만, test 위주)
- [x] GPU에서 classifier head 학습 (5 epoch, `classifier_head_gpu_v2`)
- [x] GPU에서 fine-tuning (마지막 feature block 2개, 15 epoch, `fine_tune_gpu_v2`)
- [x] real 사진 포함 test 평가 완료 (위 핵심 수치 참고)
- [x] FastAPI `/api/v1/classify`를 `fine_tune_gpu_v2` checkpoint로 연결 확인
- [ ] confidence 임계값 0.60을 real 성능 기준으로 재검토 (지금은 임시값 유지 중)

체크포인트:

```text
runs/vision/classifier_head_gpu_v2/best_model.pth
runs/vision/fine_tune_gpu_v2/best_model.pth   <- 현재 최종 후보, backend 기본값
```

옛 `classifier_head_gpu`/`fine_tune_gpu`(v1, `jub002084` 기준)는 더 이상 쓰지 않는다.

## 2. 이어서 할 작업

1. confidence 임계값(0.60) 조정 필요한지 검토 — real 오분류 1건이 0.28로 걸러진 건
   확인했지만, 표본이 54장뿐이라 더 모이면 재확인
2. 프론트엔드를 실제 checkpoint 기반 `/api/v1/classify`로 계속 연결해 실사용 테스트
3. 분류된 `artifact_id`를 RAG 설명 API(`/api/v1/docent/description`)로 연결
4. real 사진을 더 확보해 val/test 표본을 키우고, 특히 오분류가 나온 `duk000798`(청동촛대)
   각도/조명을 다양화해서 재확인
