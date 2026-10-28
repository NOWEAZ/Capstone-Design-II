# Fed-ISIC2019 실험 결과

FLamby의 Fed-ISIC2019(top3 모드: 클라이언트 3개, 8개 클래스)에서 FedAvg, FedProx, Centralized를 비교한 결과입니다.
코드는 이 브랜치(`fix/validation-split-and-metrics`)의 `main.py`, `federated_core.py`, `data_utils.py`입니다.

## 실험 개요
- 모델 EfficientNet-B0, 20라운드, batch 32, lr 1e-4. 연합 방법은 local epoch 2, Centralized는 라운드당 1 epoch.
- 데이터 분할: 클라이언트별 train 8937 / 2846 / 2423, val 1578(val_fraction=0.1, split_seed=0), test 3946.
- 라운드별 모델 선택과 지표 기록은 validation만 사용하고, test는 학습 종료 후 run마다 1회(`--final_test`)만 평가했습니다.
- 실행 환경: Colab(T4)과 Kaggle(T4 x2)에서 같은 코드로 실행했습니다. Stage 1과 시드 42 Centralized는 Colab, 나머지 Stage 2 8개 run은 Kaggle입니다.

## 단계
- **Stage 1** (`runs/stage1/`): 탐색 시드 41에서 FedAvg(μ=0)와 FedProx μ ∈ {0.001, 0.01, 0.1, 1.0}를 비교. 마지막 3라운드 Val Macro-F1 평균이 가장 높은 FedProx μ=0.001을 선택(사전 규칙). 요약은 `summary/stage1_mu_search_seed41.csv`.
- **Stage 2** (`runs/stage2/`): 시드 42, 100, 2026에서 Centralized, FedAvg, FedProx(μ=0.001) 각 1회, 총 9개 run. 요약은 `summary/stage2_test_results.csv`.

## 파일 설명
| 경로 | 내용 |
|---|---|
| `Fed-ISIC2019_results_full.pdf` | 전체 결과 보고서(표, 그림, 해석, 한계) |
| `summary/stage1_mu_search_seed41.csv` | Stage 1 μ 탐색 요약 |
| `summary/stage2_test_results.csv` | Stage 2 시드별 Test 결과(마지막 라운드 모델) |
| `runs/stage1/<run>_metrics.csv` | Stage 1 run별 라운드별 train/val 지표, 클라이언트별 F1, 누적 통신량 |
| `runs/stage2/<run>_metrics.csv` | Stage 2 run별 라운드별 지표 |
| `runs/stage2/<run>_final_test.json` | 마지막 라운드 모델(`final`)과 best-val 체크포인트(`best_val`)의 test 결과 |
| `runs/stage2/<run>_final_test_preds.npz` | test 예측(`y_true`, `y_pred`, `y_prob`, `client_idx`). 이미지는 포함하지 않음 |
| `summary/flamby_official_isic2019_summary.csv` | FLamby 공식 벤치마크(6클라이언트, 5시드) balanced accuracy 집계. 비교용 참고값 |
| `graphs/` | 학습 곡선, 혼동행렬, 통신량 그림(PNG)과 표(CSV). `graphs/data/`는 Stage 1 그림에 쓴 부분집합 CSV |

run 이름은 `<방법>_top3_seed<시드>`입니다(예: `fedprox_mu0.001_top3_seed42`).
클래스 순서는 MEL, NV, BCC, AK, BKL, DF, VASC, SCC로 가정했습니다(FLamby 순서; NV가 test에서 최다라는 점과 일치).

## FLamby 공식 결과(참고용)
`summary/flamby_official_isic2019_summary.csv`는 FLamby 저장소의 `flamby/results/results_benchmark_fed_isic2019/isic_seed42.csv`~`isic_seed46.csv`(https://github.com/owkin/FLamby)를 읽어 방법별로 집계한 값입니다.
시드 5개의 평균 ± 표본 표준편차이고, `Pooled Test`는 6개 센터 전체 test, `mean_c012`는 client_test_0~2의 평균입니다(원본 CSV는 포함하지 않았고, 논문 그림의 값과 같은지는 확인하지 않았습니다).
우리 실험(클라이언트 3개, 다른 학습 설정, 3개 센터 test 합산)과는 조건이 달라 직접 비교하면 안 됩니다.

## 포함하지 않은 것
- 모델 체크포인트(`.pt`): 용량 때문에 저장소에 넣지 않았고(`.gitignore`의 `*.pt`), 팀 Drive의 `CDII_results` 폴더에 있습니다.
- ISIC 원본 이미지와 전처리 데이터: 이미지별 라이선스 때문에 포함하지 않습니다. FLamby의 스크립트로 내려받아 전처리합니다.

## 한계
- 시드가 3개뿐이라 표준편차 추정이 거칠고, FedAvg와 FedProx(μ=0.001)의 차이는 시드 간 변동보다 작아 우열을 결론 내릴 수 없습니다.
- Stage 1(μ 선택)은 단일 시드(41)입니다.
- 지표 `macro_f1`은 실제 존재하는 클래스만으로 계산한 값(present-class), `macro_f1_fixed8`은 8개 클래스 고정 기준입니다.
- 일반 accuracy는 계산하지 않았고, 보고한 Bal Acc는 균형 정확도(클래스별 recall 평균)입니다.
