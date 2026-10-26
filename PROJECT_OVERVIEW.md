# 🏥 Fed-ISIC2019 의료 영상 연합학습 프로젝트 개요 (Project Overview)

## 1. 프로젝트 목적
본 프로젝트는 다기관 피부병변 분류 벤치마크인 **FLamby Fed-ISIC2019 (8-Class)** 환경에서 발생하는 **병원 간 비독립·비동등 분포(Non-IID: Feature Shift & Label Imbalance)** 문제를 분석하고, 표준 연합학습인 **FedAvg**와 근접 정규화 기반 **FedProx**, 그리고 상한 기준선인 **Centralized** 학습을 다각도로 비교·검증하기 위해 설계되었습니다.

단순 분류 정확도 비교에 그치지 않고, 실제 의료 AI 임상 적용에서 필수적인 다음 5가지 축을 종합 평가합니다:
1. **불균형 분류 성능**: `Balanced Accuracy`, `Macro F1-Score`
2. **신뢰도 보정(Calibration)**: 15개 구간 기준 `ECE (Expected Calibration Error)`
3. **병원 간 공정성(Client Fairness)**: 클라이언트 간 `최고-최저 Macro-F1 격차 (macro_f1_gap)`
4. **클라이언트 드리프트(Client Drift)**: 글로벌-로컬 모델 간 `L2 가중치 발산도 (weight_divergence)`
5. **통신 효율성(Communication Efficiency)**: 목표 성능 도달까지의 `누적 통신량 (Communication-to-Accuracy, MB)`

---

## 2. 워크스페이스 파일 및 모듈 구조

| 파일명 | 구분 | 주요 함수 및 역할 |
| :--- | :---: | :--- |
| **`requirements.txt`** | 환경 설정 | 필수 패키지 명세 (`torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`) |
| **`data_utils.py`** | 데이터 파이프라인 | • `get_dataloaders`: `top3`/`merge6` 모드별 3개 클라이언트 학습 로더, Centralized 로더, **Validation(Global/클라이언트별)** 로더, Test(Global/클라이언트별) 로더 생성. 센터별 FLamby train을 `stratified_train_val_split`으로 **층화 90/10 분할**(검증은 학습용 증강 없이 test와 같은 변환)<br>• `compute_class_weights`: 결측 클래스를 고려한 역빈도(Inverse-Frequency) 클래스 가중치 계산<br>• `_seed_worker`: 멀티프로세싱 DataLoader 워커 난수 시드 고정 |
| **`federated_core.py`** | 모델 및 연합학습 코어 | • `build_efficientnet_b0`: ImageNet 사전학습 `EfficientNet-B0` 로드 (`BatchNorm2d` 유지, 8클래스 분류기 교체)<br>• `train_one_client`: `FedAvg`($\mu=0$) 및 `FedProx`($\mu>0$) 로컬 학습 (순수 CE 손실 반환, 배치 크기 1 방어)<br>• `server_aggregate`: 클라이언트 샘플 수 비례 가중 평균 집계<br>• `evaluate_model`: 단일 패스 추론으로 Global 지표 및 병원별 공정성 지표(`macro_f1_gap`) 동시 산출. Macro-F1은 정답에 **존재하는 클래스만** 평균(주 지표)하고 8클래스 고정 값(`macro_f1_fixed8`)도 함께 산출. 클라이언트별 F1 반환<br>• `compute_ece`: 15-bin Expected Calibration Error 계산<br>• `compute_weight_divergence`: 글로벌-로컬 모델 간 L2 거리 계산<br>• `compute_model_size_mb` / `compute_round_comm_mb`: 모델 크기 및 라운드당 양방향 통신량(MB) 계산 |
| **`main.py`** | 학습 실행기 (CLI) | • `parse_args` / `set_seed`: CLI 인자 파싱 및 전체 난수(Python, NumPy, PyTorch, CUDA) 고정<br>• 매 라운드 **Validation** 지표 실시간 CSV 로깅, 최고 Val 모델(`best_*.pt`) 및 세션 중단 복구용 체크포인트(`last_*.pt`, `--resume`) 원자적 저장<br>• `--final_test`: 학습 종료 후 FLamby **test를 1회만** 평가(`*_final_test.json`, 예측 `*_final_test_preds.npz`) |
| **`fed_isic2019_colab.ipynb`** | 협업 실행 노트북 | • Phase 0~1: GPU 확인, Drive 마운트, FLamby 설치, ISIC 데이터 전처리(`tar` 캐싱) 및 Smoke Test<br>• Phase 2~4: **2단계 시드 전략**(Stage 1 `seed=42` 탐색 $\rightarrow$ Stage 2 `seed=100, 2026` 추가 실행)<br>• Phase 5: **Final(주 결과) / Best(참고) `Mean ± Std` 요약표**, 목표 정확도 도달 통신량 집계표, 학습 곡선 및 통신 효율성 그래프 생성 |
| **`DESIGN_DECISIONS_AND_CHANGELOG.md`** | 기술 문서 | • 아키텍처·평가지표·하이퍼파라미터 의사결정 및 1~3차 코드 감사 내역과 학술적·엔지니어링 근거 정리 |

---

## 3. 데이터셋 분할 모드 (`--mode`)

FLamby의 `FedIsic2019`는 총 6개 병원/센터(Center 0 ~ 5, 총 학습 18,597장 / 테스트 4,650장)로 구성되며, 본 프로젝트에서는 3개 클라이언트 환경에서 두 가지 분할 모드를 지원합니다.

| 모드 | 클라이언트 구성 | 포함된 원본 센터 (데이터 출처) | 학습 샘플 수 (Train) | 테스트 샘플 수 (Test) |
| :---: | :---: | :--- | :---: | :---: |
| **`top3`** | **Client 0** | Center 0 (BCN) | 9,930 | 2,483 |
| *(상위 3개 대형 병원)* | **Client 1** | Center 1 (HAM10000 - ViDIR MoleMax) | 3,163 | 791 |
| | **Client 2** | Center 2 (HAM10000 - ViDIR Modern) | 2,691 | 672 |
| | **합계 (Global)** | **Center 0, 1, 2 통합** | **15,784** | **3,946** |
| **`merge6`** | **Client 0** | Center 0 (9,930) + Center 3 (MSK, 1,807) | 11,737 | 2,935 |
| *(전체 6개 병원 3분할 병합)* | **Client 1** | Center 1 (3,163) + Center 4 (Rosendahl, 655) | 3,818 | 955 |
| | **Client 2** | Center 2 (2,691) + Center 5 (ViDIR Legacy, 351) | 3,042 | 760 |
| | **합계 (Global)** | **Center 0 ~ 5 전체 통합** | **18,597** | **4,650** |

> **평가 도메인 정합성**: Global Validation/Test 셋은 각 모드에 참여한 클라이언트의 검증/테스트셋만 통합(`ConcatDataset`)하여 학습 도메인과 평가 도메인을 엄밀하게 일치시킵니다.

> **Train / Validation / Test 구성**: 위 표의 Train·Test 수는 FLamby 공식 분할입니다. 이 중 센터별 **Train의 10%를 층화로 떼어 Validation**으로 쓰고(`--val_fraction 0.1`, 분할 시드 `--split_seed 0` 고정) 나머지 90%만 학습합니다. 매 라운드 평가, 최고 라운드 저장, $\mu$ 선택은 모두 Validation으로 하고, **Test는 학습이 끝난 뒤 1회만** 평가합니다(`--final_test`).

---

## 4. 모델 아키텍처 및 학습 기법 (`--method`)

### 4.1 백본 모델 및 클래스 불균형 가중 손실 함수
- **모델**: `torchvision.models.efficientnet_b0(pretrained=True)`
  - 사전학습 특징 분포 보존을 위해 내부 `BatchNorm2d`를 유지하고, 최종 분류기(`classifier[1]`)만 `nn.Linear(1280, 8)`로 교체합니다.
- **역빈도(Inverse-Frequency) 가중 Cross-Entropy**:
  - 각 클라이언트의 로컬 클래스 샘플 수 $n_c$에 반비례하는 가중치를 부여하되, 일부 병원에 샘플이 `0`개인 결측 클래스가 있더라도 유효 학습률 스케일이 흔들리지 않도록 **존재하는 클래스($n_c > 0$)들의 가중치 평균이 `1.0`** 이 되도록 정규화합니다:
    $$w_c = \begin{cases} \dfrac{N}{K \cdot n_c} & (n_c > 0) \\ 0 & (n_c = 0) \end{cases}, \qquad \tilde{w}_c = \frac{w_c}{\frac{1}{|\{j : n_j > 0\}|} \sum_{j : n_j > 0} w_j}$$

### 4.2 비교 학습 기법 (3종)
1. **`centralized` (상한 기준선)**:
   - 모든 참여 클라이언트의 학습 데이터를 하나로 모아 단일 글로벌 모델을 학습합니다 (`rounds = 20`, `local_epochs = 1`, 에폭 간 Adam 모멘텀 유지).
2. **`fedavg` ($\mu = 0.0$)**:
   - 각 클라이언트가 로컬 데이터로 `local_epochs = 2` 동안 학습한 뒤, 서버에서 클라이언트 샘플 수($N_k$)에 비례한 가중 평균으로 글로벌 모델을 업데이트합니다:
     $$w_{\text{global}}^{(t+1)} = \sum_{k=1}^{K} \frac{N_k}{N} w_k^{(t+1)}$$
3. **`fedprox` ($\mu > 0.0$)**:
   - 로컬 학습 목적함수에 글로벌 모델 $w_{\text{global}}^{(t)}$로부터의 이탈을 억제하는 근접 정규화 항(Proximal Term)을 추가하여 Non-IID 환경의 **Client Drift**를 완화합니다 (`local_epochs = 2`):
     $$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{CE}}(w) + \frac{\mu}{2} \left\| w - w_{\text{global}}^{(t)} \right\|_2^2$$
   - 단, CSV에 기록되는 `train_loss`는 FedAvg와 동일 선상에서 비교할 수 있도록 근접 항을 제외한 순수 $\mathcal{L}_{\text{CE}}(w)$만 기록합니다.

---

## 5. 기록 및 평가 지표 (CSV 컬럼 명세)

매 라운드 종료 시 `--save_dir` 내 `{exp_tag}_seed{seed}_metrics.csv`에 실시간 기록되는 컬럼입니다. **모든 지표는 Validation 기준**(`eval_split=val`)입니다. Test 결과는 `{exp_tag}_seed{seed}_final_test.json`에 별도로 저장됩니다.

| 컬럼명 | 의미 및 해석 방법 | 우수 방향 |
| :--- | :--- | :---: |
| `round` | 현재 통신 라운드 번호 (`1 ~ rounds`) | - |
| `method` / `mode` / `mu` / `seed` | 실험 식별 메타데이터 (`centralized`/`fedavg`/`fedprox`, `top3`/`merge6`, $\mu$, 난수 시드) | - |
| `eval_split` | 이 행의 지표를 계산한 평가 셋 (항상 `val`) | - |
| **`train_loss`** | 학습 데이터의 샘플 수 가중 평균 순수 Cross-Entropy 손실 | $\downarrow$ |
| **`val_loss`** | Global Validation 셋의 가중 Cross-Entropy 손실 | $\downarrow$ |
| **`balanced_acc`** | 8개 클래스별 재현율(Recall)의 산술 평균 (**핵심 분류 성능 지표** 및 `best_*.pt` 저장 기준) | $\uparrow$ |
| **`macro_f1`** | Global Validation 셋 정답에 존재하는 클래스들의 F1-Score 비가중 평균 | $\uparrow$ |
| `macro_f1_fixed8` | 8개 클래스 전부(없는 클래스는 F1=0)로 평균한 보조 Macro-F1 | $\uparrow$ |
| **`ece`** | 15개 구간(`n_bins=15`) 기준 **Expected Calibration Error** (**모델 신뢰도 보정 지표**) | $\downarrow$ |
| **`max_macro_f1` / `min_macro_f1`** | 3개 클라이언트 로컬 검증셋 중 가장 높은 / 가장 낮은 `Macro-F1`(존재 클래스 기준) | $\uparrow$ |
| `macro_f1_client0~2` / `macro_f1_fixed8_client0~2` | 클라이언트(병원)별 Macro-F1 (존재 클래스 기준 / 8클래스 고정). 어느 병원이 약한지 확인용 | $\uparrow$ |
| **`macro_f1_gap`** | 클라이언트 간 최고-최저 Macro-F1 격차 ($\max_k \text{F1}_k - \min_k \text{F1}_k$, **병원 간 공정성 지표**) | $\downarrow$ |
| **`weight_divergence`** | 글로벌 모델과 로컬 모델 파라미터 간 평균 L2 거리 ($\|w_{\text{local}} - w_{\text{global}}\|_2$, **Client Drift 지표**) | 분석용 |
| **`cumulative_comm_mb`** | 해당 라운드까지의 누적 양방향 통신량(MB). (1라운드당 `2 × 3 × 15.3 MB ≈ 91.9 MB`, Centralized는 `0`) | $\downarrow$ |

---

## 6. CLI 인자 및 실행 방법

### 6.1 `main.py` 지원 인자 목록
| 인자 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--method` | `fedavg` | 학습 방식 선택: `centralized`, `fedavg`, `fedprox` |
| `--mode` | `top3` | 데이터셋 분할 모드: `top3`, `merge6` |
| `--rounds` | `20` | 총 통신 라운드 수 |
| `--local_epochs` | `2` | 라운드당 로컬 에폭 수 (연합학습 권장: `2`, Centralized 권장: `1`) |
| `--batch_size` | `32` | 미니배치 크기 |
| `--lr` | `1e-4` | Adam 옵티마이저 학습률 |
| `--mu` | `0.01` | FedProx 근접 항 계수 (`fedavg`/`centralized`에서는 자동으로 `0.0` 적용) |
| `--seed` | `42` | 재현성 및 다중 시드 실험용 난수 시드 (권장 3시드: `42, 100, 2026`) |
| `--num_workers` | `4` | DataLoader CPU 워커 프로세스 수 (코랩 기본 런타임 권장: `2`) |
| `--data_path` | `None` | 전처리된 `ISIC_2019_Training_Input_preprocessed` 폴더가 위치한 상위 디렉토리 경로 |
| `--save_dir` | `./results` | 결과 CSV 및 모델 체크포인트(`.pt`) 저장 경로 |
| `--resume` | `False` (플래그) | 동일 실험의 `last_*.pt`가 존재할 경우 마지막 완료 라운드 다음부터 이어서 학습 |
| `--val_fraction` | `0.1` | 센터별 FLamby train에서 층화로 떼어 낼 Validation 비율 |
| `--split_seed` | `0` | train/val 분할 시드 (실험 시드와 무관하게 고정) |
| `--final_test` | `False` (플래그) | 학습 종료 후 마지막 라운드 모델(과 best-val 모델)을 test로 **1회** 평가해 저장. 결과 파일이 이미 있으면 다시 열지 않음 |

### 6.2 CLI 실행 예시
```bash
# 1. Centralized Baseline (20 라운드, 라운드당 1 에폭)
python main.py --method centralized --mode top3 --rounds 20 --local_epochs 1 --batch_size 32 --lr 1e-4 --seed 42 --data_path ./isic_data --save_dir ./results --resume

# 2. FedAvg (20 라운드, 라운드당 로컬 2 에폭)
python main.py --method fedavg --mode top3 --rounds 20 --local_epochs 2 --batch_size 32 --lr 1e-4 --mu 0.0 --seed 42 --data_path ./isic_data --save_dir ./results --resume

# 3. FedProx (mu=0.01, 20 라운드, 라운드당 로컬 2 에폭)
python main.py --method fedprox --mode top3 --rounds 20 --local_epochs 2 --batch_size 32 --lr 1e-4 --mu 0.01 --seed 42 --data_path ./isic_data --save_dir ./results --resume
```

---

## 7. Google Colab 단계별 실행 프로토콜 (`fed_isic2019_colab.ipynb`)

1. **Phase 0 & Phase 1 (환경 및 데이터 준비)**:
   - Google Drive를 마운트하고 FLamby(`pip install -e ".[isic2019]"`)를 설치합니다.
   - 전처리된 이미지(23,247장)는 최초 1회만 생성하여 Drive에 `isic2019_preprocessed.tar`로 보관하고, 이후 세션에서는 로컬 디스크(`/content/isic_data`)로 빠르게 압축 해제하여 학습 I/O 속도를 확보합니다.
2. **Phase 2 & Phase 3 (Stage 1 — 탐색 시드 `seed=41`로 $\mu$ 선택)**:
   - 탐색 전용 시드 `41`로 `FedAvg`($\mu=0$ 기준선)와 `FedProx`의 $\mu$ 후보군(`0.001, 0.01, 0.1, 1.0`)을 1회씩 실행합니다. 비교는 **Validation** 기준 마지막 3라운드 평균으로 하고, 최적의 `BEST_MU`는 결과를 보고 **직접 지정**합니다. 이 시드의 결과는 최종 표에 포함하지 않습니다.
3. **Phase 4 (Stage 2 — 최종 3시드 검증)**:
   - 선정된 3가지 모델(`Centralized`, `FedAvg`, `FedProx(BEST_MU)`)을 최종 시드 `42, 100, 2026` 3개로 실행하고, 각 실행 종료 시 `--final_test`로 test를 1회만 평가합니다.
   - 코랩 세션이 중간에 끊기더라도 `--resume` 옵션 덕분에 중단된 셀을 다시 실행하면 마지막 완료 라운드 다음부터 즉시 재개됩니다.
4. **Phase 5 (최종 요약표 및 그래프 자동 생성)**:
   - `summary_final_mean_std_{MODE}.csv`: 마지막 라운드(Final) 기준 3시드 `Mean ± Std` 요약표 (Validation 기준)
   - `summary_best_mean_std_{MODE}.csv`: **최고 Val Balanced Acc 라운드(Best) 기준 3시드 `Mean ± Std` 요약표 (참고용)**
   - `summary_test_final_mean_std_{MODE}.csv` / `summary_test_best_val_mean_std_{MODE}.csv`: `--final_test`로 1회 평가한 **Test `Mean ± Std` 요약표** (마지막 라운드 모델이 보고서 주 결과, best-val 모델은 참고)
   - `comm_to_target_{MODE}.csv`: 목표 정확도 도달 비율, 평균 도달 라운드, 평균 누적 통신량(MB) 비교표
   - `comparison_curves_mean_std_{MODE}.png` & `comm_efficiency_{MODE}.png`: 4대 지표 라운드별 추이 그래프 및 누적 통신량 대비 정확도 효율성 그래프 (`±1 Std` 음영 포함)
