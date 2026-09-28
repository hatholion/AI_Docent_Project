# Vision 학습 현황 및 다음 작업

이 노트북 자체에 GPU(RTX 4060 Laptop)가 있어서, 별도 GPU 노트북으로 옮기는 절차 없이
이 문서에 있던 clone/환경 이전 계획은 더 이상 필요하지 않다. 환경 설치·학습·평가 명령은
전부 `README.md`의 "Vision 학습 환경", "분류 API 실행" 절에 정리되어 있으니 여기서는
중복 기재하지 않고, 현재 상태와 다음 작업만 남긴다.

## 1. 완료된 상태

- [x] 7개 유물 클래스 메타데이터 등록 (`bon002789`, `duk000798`, `jub002084`, `ssu001794`, `ssu001846`, `ssu003094`, `ssu022891`)
- [x] synthetic 이미지 1,320장 train/val/test 분할
- [x] GPU에서 classifier head 학습 (5 epoch) → test accuracy 1.0, macro F1 1.0
- [x] GPU에서 fine-tuning (마지막 feature block 2개, 15 epoch) → test accuracy 1.0, macro F1 1.0
- [x] FastAPI `/api/v1/classify` 구현 및 checkpoint 연결 확인
- [ ] 실제 핸드폰 촬영 데이터 수집 및 평가

체크포인트:

```text
runs/vision/classifier_head_gpu/best_model.pth
runs/vision/fine_tune_gpu/best_model.pth   <- 현재 최종 후보
```

## 2. 데이터 현황 (최종, synthetic 기준)

| artifact_id | 유물명 | train | val | test |
|---|---|---:|---:|---:|
| `bon002789` | 금동 반가사유상 | 140 | 30 | 30 |
| `duk000798` | 청동촛대 | 140 | 30 | 30 |
| `jub002084` | 백자 달항아리 | 119 | 26 | 25 |
| `ssu001794` | 농경문 청동기 | 140 | 30 | 30 |
| `ssu001846` | 방패형 동기 | 140 | 30 | 30 |
| `ssu003094` | 요령식 동검 | 140 | 30 | 30 |
| `ssu022891` | 빗살무늬토기 | 105 | 22 | 23 |
| **합계** |  | **924** | **198** | **198** |

## 3. 주의사항

현재 train/val/test가 전부 같은 Blender synthetic 렌더링이라 test accuracy가 1.0으로 나온다.
이는 synthetic 렌더링 특징을 구분한 결과일 수 있으며, 실제 핸드폰 촬영 사진에 대한 서비스
성능을 의미하지 않는다. 실제 데이터로 별도 검증하기 전까지는 이 수치를 최종 성능으로
사용하지 않는다.

## 4. 이어서 할 작업

1. 실제 핸드폰 사진 확보 (유물당 15~30장, 각도/조명/배경 다양하게; 직접 촬영이 어려우면
   인터넷 사진 사용 가능하나 저작권 확인 및 출처 기록 필요)
2. 촬영 세션 단위로 val/test에 분리 배치 (같은 세션 사진이 val/test에 나뉘어 들어가지
   않도록 하여 data leakage 방지)
3. synthetic와 real 성능을 별도로 평가
4. real validation 결과로 confidence 임계값(현재 0.60은 임시값) 확정
5. 프론트엔드 업로드 화면을 `/api/v1/classify`에 연결
6. 분류된 `artifact_id`를 RAG 설명 API(`/api/v1/docent/description`)로 연결
