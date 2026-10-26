# 🛠️ 설계 의사결정 및 코드 수정 내역서 (Design Decisions & Changelog)

본 문서는 Fed-ISIC2019 연합학습 프로젝트를 구축하고 고도화하는 과정에서 논의된 **6가지 핵심 기술적 의사결정 사항, 코드 반영 내역, 그리고 학술적·실험적 근거**를 정리한 문서입니다. 보고서 및 논문 작성 시 실험 설계 근거(Experimental Setup & Justification)로 바로 활용할 수 있습니다.

---

## 1. 정규화 계층 선택: `GroupNorm(2 groups)` 대신 `BatchNorm` 유지

### 📌 검토 내용
- Non-IID 연합학습 환경에서 병원별 배치 통계량(`running_mean`, `running_var`) 충돌을 피하기 위해 `EfficientNet-B0` 내부의 `BatchNorm2d`를 `GroupNorm(num_groups=2)`로 교체하는 방안을 검토함.

### ✅ 최종 결정
- **기존 `BatchNorm2d`를 그대로 유지하기로 결정.**

### 💡 기술적 근거
1. **사전학습(Pretrained) 가중치 분포 보존**:
   - `build_efficientnet_b0(pretrained=True)`로 로드하는 ImageNet 사전학습 합성곱(Conv) 가중치들은 기존 `BatchNorm2d`의 채널별 정규화 스케일을 전제로 학습되어 있음.
   - 이를 `GroupNorm`으로 교체하면 중간 활성화(Activation) 분포가 급격히 틀어져 **초기 라운드에서 사전학습된 특징 추출 능력이 붕괴(Representation Collapse)** 되며, 제한된 통신 라운드(20 라운드) 내에 수렴하지 못하는 손실이 발생함.
2. **`num_groups=2`의 채널 해상도 한계**:
   - `EfficientNet-B0`의 뒷단 채널 수는 `320`, `1280`까지 커지는데, 이를 단 2개 그룹으로 묶으면 사실상 `LayerNorm`(`num_groups=1`)에 가까워져 개별 채널의 미세한 병변 특징 스케일이 뭉개짐.
   - 따라서 제한된 라운드 내 최고 `Balanced Accuracy` 확보를 위해 Pretrained `BatchNorm2d` 구조를 유지하는 것이 가장 합리적임.

---

## 2. 평가지표 고도화: `ECE` 및 `최고-최저 Macro-F1 격차` 추가

### 📌 수정 내역
- **기존 지표**: `loss`, `balanced_acc`, `macro_f1`, `weight_divergence`
- **추가 지표**:
  1. **`ece` (Expected Calibration Error)** (`federated_core.py`의 `compute_ece` 함수)
  2. **`max_macro_f1`, `min_macro_f1`, `macro_f1_gap`** (`data_utils.py`에서 클라이언트별 테스트 로더 `test_loader.client_test_loaders`를 연결하고 `evaluate_model`에서 산출)

### 💡 기술적 근거
1. **의료 AI에서의 신뢰도 보정(Calibration) 중요성 (`ECE`)**:
   - 피부암 진단 보조 시스템에서는 예측 클래스뿐만 아니라 **"모델이 90% 확률로 확신할 때 실제 정답률도 90%인가?"** 를 뜻하는 신뢰도 보정이 필수적임.
   - Non-IID 연합학습에서는 각 병원의 로컬 에폭이 반복될수록 특정 병원 분포에 과신(Over-confidence)하여 ECE가 악화되기 쉬우며, FedProx가 이를 완화하는지 검증할 수 있음.
2. **병원 간 공정성(Client Fairness) 정량화 (`macro_f1_gap`)**:
   - 데이터 수가 많은 대형 병원(예: Center 0)에서만 성능이 높고 소형 병원에서 성능이 낮다면 연합학습의 실효성이 떨어짐.
   - 3개 클라이언트 각각의 로컬 테스트셋에서 계산한 `Macro-F1`의 **최고값과 최저값의 차이($\max_k \text{F1}_k - \min_k \text{F1}_k$)** 를 측정하여 병원 간 성능 편차(불공정성)가 얼마나 줄어드는지 직접 평가함.

---

## 3. ECE 구간 수 설정: `n_bins=10` 대신 `n_bins=15` 채택

### 📌 검토 내용
- ECE 계산 시 신뢰도 구간 `[0.0, 1.0]`을 10개 구간(`n_bins=10`)으로 나눌지, 15개 구간(`n_bins=15`)으로 나눌지 검토함.

### ✅ 최종 결정
- **`n_bins=15` 유지.**

### 💡 기술적 근거
1. **학계 및 라이브러리 표준(De facto Standard)**:
   - 현대 신경망 Calibration 연구의 기준 논문인 **Guo et al., *"On Calibration of Modern Neural Networks"* (ICML 2017)** 과 PyTorch 공식 지표인 `torchmetrics.CalibrationError` 모두 기본값으로 `n_bins=15`를 사용함.
2. **8클래스 분류의 고신뢰 구간 해상도(Resolution) 확보**:
   - 8클래스 분류에서 최대 소프트맥스 확률값($\max_c \hat{p}_c$)의 이론적 하한은 $\frac{1}{8} = 0.125$이며, 학습이 진행될수록 예측 확률이 `[0.7, 1.0]` 구간에 밀집됨.
   - `n_bins=10`(구간 폭 `0.1`)은 고신뢰 구간을 너무 넓게 묶어 과신(Over-confidence) 오차를 둔감하게 만드는 반면, `n_bins=15`(구간 폭 약 `0.067`)는 고신뢰 영역을 더 세밀하게 쪼개어 보정 오차를 정밀하게 포착함. (Global Test 샘플 수가 약 4,600개 이상이므로 15개 구간에서도 표본 부족 문제가 발생하지 않음.)

---

## 4. 다중 시드 실험 전략: `방식 B (1시드 탐색 -> 최종 3시드 검증)` 도입

### 📌 수정 내역
- `main.py`: `set_seed(seed)` 함수 및 `--seed` 인자 추가, 파일명에 `_seed{seed}` 및 FedProx의 `_mu{mu}` 자동 표기.
- `data_utils.py`: `get_dataloaders`에 `seed` 인자, `_seed_worker`, `torch.Generator` 적용.
- `fed_isic2019_colab.ipynb`:
  - **Stage 1 (`EXPLORE_SEED = 42`)**: 단일 시드로 `Centralized`, `FedAvg`, `FedProx`($\mu$ 후보군)를 1회씩 빠르게 탐색.
  - **Stage 2 (`EXTRA_SEEDS = [100, 2026]`)**: 확정된 최종 모델에 대해서만 나머지 2개 시드를 추가 실행해 총 3시드(`42, 100, 2026`) 완성 및 `Mean ± Std` 요약표·표준편차 음영 그래프 생성.

### 💡 기술적 근거
1. **통계적 유의성 확보**:
   - 새로 교체한 8클래스 분류기(`classifier[1]`)의 가중치 초기화와 미니배치 셔플 순서에 따른 우연한 노이즈를 배제하고, FLamby 공식 벤치마크와 동일하게 `평균 ± 표준편차 (Mean ± Std)`로 결과를 제시하기 위함.
2. **Google Colab 런타임 제약 극복 (탐색 비용 절감)**:
   - 모든 $\mu$ 후보군까지 처음부터 3개 시드로 돌리면 코랩 일일 GPU 할당량을 초과하기 쉬움.
   - `seed=42`로 먼저 최적의 $\mu^*$를 찾고, 최종 비교 대상에 대해서만 이미 완료된 `seed=42` 결과에 `seed=100, 2026` 두 번의 실험만 보태어 **실험 시간을 40% 이상 절약하면서도 완벽한 3시드 검증**을 달성함.

---

## 5. 통신 라운드 및 로컬 에폭 조정: `대안 2 (rounds=20, 연합학습 local_epochs=2)`

### 📌 수정 내역
- `main.py` 및 `fed_isic2019_colab.ipynb`:
  - 총 통신 라운드 수: `ROUNDS = 20` 유지
  - **Centralized** 로컬 에폭: `CENTRAL_LOCAL_EPOCHS = 1` (총 20 에폭)
  - **FedAvg / FedProx** 로컬 에폭: `FL_LOCAL_EPOCHS = 2` (각 클라이언트 누적 40 로컬 에폭 효과)

### 💡 기술적 근거
1. **연합학습의 가중치 희석 보완 및 수렴 보장**:
   - `batch_size=32` 기준 Client 1, 2의 1 에폭은 약 99 스텝에 불과함. 연합학습은 매 라운드 서버 집계 시 서로 다른 병원 가중치가 평균화되면서 학습 방향이 희석되므로, `local_epochs=1`로 20 라운드만 돌리면 충분히 수렴하지 못한 채 종료될 위험이 있음.
   - `local_epochs=2`로 설정하면 테스트셋 평가(Evaluation) 오버헤드는 20회로 동일하게 아끼면서도 실질적인 로컬 학습량을 40 에폭으로 확보해 20 라운드 내 안정적 수렴을 보장함.
2. **FedAvg vs FedProx 변별력(Client Drift) 극대화**:
   - `local_epochs=1`에서는 로컬 업데이트 스텝이 너무 짧아 각 병원의 가중치가 글로벌 모델에서 멀어질 틈이 없으므로 FedAvg와 FedProx 차이가 미미함.
   - `local_epochs=2`로 로컬 학습 스텝을 늘리면 병원 간 **Client Drift(`weight_divergence`)가 명확히 유발**되어, 이를 억제하는 **FedProx의 정확도 방어 및 공정성(`macro_f1_gap`) 개선 효과가 뚜렷하게 관찰**됨.

---

## 6. 통신 효율성 비교 지표 도입: `누적 통신량(MB) 및 목표 성능 도달 통신 비용`

### 📌 수정 내역
- `federated_core.py`: `compute_model_size_mb`, `compute_round_comm_mb` 함수 추가.
- `main.py`: 매 라운드 양방향 누적 통신량(`cumulative_comm_mb`) 계산 및 CSV 기록.
- `fed_isic2019_colab.ipynb`: 목표 정확도(`TARGET_ACC`) 최초 도달 시점의 라운드 수(`Rounds to Target`) 및 누적 통신량(`Comm to Target (MB)`) 비교표와 통신 효율성 곡선 그래프 추가.

### 💡 기술적 근거
1. **1라운드당 단순 통신량 비교의 한계 극복**:
   - FedProx의 근접 항은 클라이언트 내부에서 계산되므로 1라운드당 송수신 데이터 크기는 FedAvg와 FedProx가 `2 × 3 × 15.3 MB ≈ 91.9 MB`로 완전히 동일함.
2. **Communication-to-Accuracy 관점의 실효성 입증**:
   - 연합학습 원 논문(McMahan et al., Li et al.)의 표준 평가 방식에 따라 **"동일한 임상 목표 정확도(Target Balanced Acc)에 도달하기까지 소모된 누적 통신량(MB)"** 을 비교함.
   - 이를 통해 FedProx가 Client Drift로 인한 수렴 지연을 줄여 **목표 성능 달성에 필요한 병원-서버 간 네트워크 통신 비용을 얼마나 절감하는지** 정량적으로 제시할 수 있음.

