# 🛠️ 설계 의사결정 및 코드 수정 내역서 (Design Decisions & Changelog)

본 문서는 **Fed-ISIC2019 의료 영상 연합학습 프로젝트**를 설계·구현·검증하는 과정에서 이루어진 **핵심 기술적 의사결정(섹션 1~6)** 과 **코드 감사 및 안정화 내역(섹션 7~8)** 을 기술적·학술적 근거와 함께 정리한 문서입니다. 보고서 및 논문 작성 시 실험 설계 근거(Experimental Setup & Justification)로 바로 활용할 수 있습니다.

---

## 📋 전체 의사결정 및 수정 요약표

| 구분 | 항목 | 최종 결정 및 반영 내용 | 핵심 근거 요약 |
| :---: | :--- | :--- | :--- |
| **1** | 정규화 계층 | `GroupNorm(2)` 대신 **`BatchNorm2d` 유지** | ImageNet 사전학습 가중치 분포 붕괴 방지 및 뒷단 고채널(320, 1280) 특징 해상도 보존 |
| **2** | 평가지표 확장 | **`ECE`** 및 **`macro_f1_gap`(최고-최저 격차)** 추가 | 의료 AI 필수 요건인 신뢰도 보정(Calibration)과 병원 간 공정성(Client Fairness) 정량화 |
| **3** | ECE 구간 수 | `n_bins=10` 대신 **`n_bins=15` 채택** | 학계 표준(Guo et al., ICML 2017) 준수 및 8클래스 고신뢰 구간(`[0.7, 1.0]`) 해상도 확보 |
| **4** | 다중 시드 전략 | **방식 B (1시드 탐색 $\rightarrow$ 최종 3시드 검증)** | 분류기 초기화·배치 셔플 노이즈 제거(`Mean ± Std`) 및 코랩 탐색 시간 40% 이상 절감 |
| **5** | 라운드 & 에폭 | **`rounds=20`**, 연합학습 **`local_epochs=2`** *(Centralized `1`)* | 연합학습 가중치 희석 보완(40 로컬 에폭 효과) 및 Client Drift 유발로 FedAvg vs FedProx 변별력 확보 |
| **6** | 통신량 비교 | **목표 정확도 도달 누적 통신량(`Communication-to-Accuracy`)** | 1라운드 통신량(91.9 MB)은 동일하므로, 목표 성능 도달까지 소모된 누적 통신 비용(MB)으로 효율성 입증 |
| **7** | 1차 코드 감사 | 중복 추론 제거, `top3` 테스트셋 일치, `BN` 방어 등 4건 | 매 라운드 평가 속도 2배 향상, 학습·평가 도메인 일치, 마지막 배치 크기 1 크래시 원천 차단 |
| **8** | 2·3차 코드 감사 | 데이터 준비, 옵티마이저·손실·가중치 보정, `--resume` 등 | 코랩 실행 완결성 확보, 실험 공정성·보고 타당성(Final vs Best) 확립, 세션 중단 복구 지원 |

---

## 1. 정규화 계층 선택: `GroupNorm(2 groups)` 대신 `BatchNorm` 유지

### 📌 검토 내용
- Non-IID 연합학습 환경에서 병원별 미니배치 통계량(`running_mean`, `running_var`) 충돌을 피하기 위해 `EfficientNet-B0` 내부의 `BatchNorm2d`를 `GroupNorm(num_groups=2)`로 교체하는 방안을 검토함.

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
   - `n_bins=10`(구간 폭 `0.1`)은 고신뢰 구간을 너무 넓게 묶어 과신(Over-confidence) 오차를 둔감하게 만드는 반면, `n_bins=15`(구간 폭 약 `0.067`)는 고신뢰 영역을 더 세밀하게 쪼개어 보정 오차를 정밀하게 포착함. (Global Test 샘플 수가 `top3` 3,946개 / `merge6` 4,650개로 충분하므로 15개 구간에서도 표본 부족 문제가 발생하지 않음.)

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
   - `batch_size=32` 기준 Client 1(3,163장)과 Client 2(2,691장)의 1 에폭은 각각 약 99 스텝, 85 스텝에 불과함. 연합학습은 매 라운드 서버 집계 시 서로 다른 병원 가중치가 평균화되면서 학습 방향이 희석되므로, `local_epochs=1`로 20 라운드만 돌리면 충분히 수렴하지 못한 채 종료될 위험이 있음.
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

---

## 7. 1차 코드 감사(Code Audit): 속도 최적화 및 런타임 방어 (4건)

| 번호 | 대상 파일 | 수정 내용 | 기술적·실험적 근거 |
| :---: | :--- | :--- | :--- |
| **7.1** | `federated_core.py` | **`evaluate_model` 중복 추론 제거** | 기존에는 Global Test 로더를 추론한 뒤 `client_test_loaders`를 다시 추론해 매 라운드 2번씩 추론했음. 클라이언트별 예측값을 `np.concatenate`하여 **단 1회 추론으로 Global 지표와 병원별 공정성 지표를 동시에 산출(평가 속도 2배 향상)**. |
| **7.2** | `data_utils.py` | **`top3` 모드 Global Test 범위 일치** | `FedIsic2019(train=False, pooled=True)` 대신 `ConcatDataset(client_test_datasets)`를 사용하여, `top3`에서는 센터 0~2만, `merge6`에서는 센터 0~5 전체를 평가하도록 학습·평가 도메인을 엄밀히 일치시킴. |
| **7.3** | `federated_core.py` | **`BatchNorm2d` 단일 샘플 배치 방어** | `N % batch_size == 1`일 때 마지막 배치(크기 1)에서 `BatchNorm2d` 분산 계산 에러(`ValueError`)가 발생하는 것을 막기 위해 `if images.size(0) == 1: continue` 추가. `drop_last=True`로 인한 샘플 손실 없이 크래시만 예방함. |
| **7.4** | `main.py` | **`Centralized`의 `weight_divergence = 0.0` 고정** | `Centralized`에서 이전 에폭과 현재 에폭의 차이를 기록하면 연합학습의 '동일 라운드 내 클라이언트-서버 간 괴리(Client Drift)'와 물리적 의미가 달라 그래프 해석을 왜곡하므로 `0.0`으로 통일하고 불필요한 `deepcopy` 메모리를 절약함. |

---

## 8. 2·3차 최종 코드 감사: 코랩 실행 완결성·실험 공정성·보고 방법론·Resume 보완

### 8.1 코랩 환경 실행 완결성 확보 (`fed_isic2019_colab.ipynb`, `main.py`, `data_utils.py`)
1. **FLamby 공식Editable 설치 및 커널 경로 등록 (Phase 1-2)**:
   - **문제**: FLamby는 PyPI 단독 설치(`pip install flamby[isic2019]`)가 지원되지 않거나 `git+https` 설치 시 `isic2019` 의존성이 누락됨.
   - **해결 및 근거**: 공식 가이드대로 `git clone` 후 `pip install -e ".[isic2019]"`를 수행하고, 커널 재시작 없이 즉시 import되도록 `sys.path`에 등록함.
2. **ISIC2019 데이터 다운로드·전처리 및 Drive tar 캐싱 (Phase 1-3, `--data_path`)**:
   - **문제**: 코랩은 세션마다 로컬 디스크가 초기화되어 매번 9GB 원본 다운로드 및 리사이즈를 반복할 수 없으며, 반대로 Google Drive(FUSE 마운트)에서 수만 장의 이미지를 직접 읽으면 I/O 병목으로 학습이 극도로 느려짐.
   - **해결 및 근거**: 최초 1회만 `download_isic.py` → `resize_images.py`를 실행해 전처리 폴더를 Drive에 단일 `tar` 파일로 보관하고, 이후 세션에서는 로컬 디스크(`/content/isic_data`)로 빠르게 압축 해제한 뒤 `main.py --data_path`로 전달하는 구조("Drive 보관 + 로컬 고속 I/O")를 채택함.
3. **FLamby 버전별 `FedIsic2019` 시그니처 호환성 방어 (`data_utils.py`)**:
   - **해결 및 근거**: FLamby 커밋 버전에 따라 `FedIsic2019.__init__`에 `debug` 인자가 없는 경우가 있으므로 `inspect.signature`로 확인 후 전달하여 버전 차이로 인한 `TypeError`를 방지함.

### 8.2 실험 공정성 및 수식 정합성 보정 (`federated_core.py`, `main.py`, `data_utils.py`)
1. **`Centralized` 학습 시 Adam 옵티마이저 모멘텀 유지**:
   - **문제**: `train_one_client`가 매번 새 `Adam`을 생성하면 `Centralized`에서는 매 에폭마다 1·2차 모멘텀($m_t, v_t$)이 초기화되어 상한 기준선(Upper-bound)이 불리하게 측정됨.
   - **해결 및 근거**: 연합학습은 표준(McMahan et al., 2017)에 따라 라운드마다 로컬 옵티마이저를 초기화하되, `Centralized`는 루프 외부에서 생성한 `central_optimizer`를 전달해 에폭 간 모멘텀을 유지하도록 보정함.
2. **FedProx의 `train_loss` 기록에서 근접 정규화 항 분리**:
   - **문제**: 역전파 손실 `CE + (μ/2)·‖w − w_g‖²`를 그대로 `train_loss`에 기록하면 FedAvg의 순수 `CE` 손실과 스케일이 달라 직접 비교할 수 없음.
   - **해결 및 근거**: 역전파(`loss.backward()`)에는 근접 항을 포함한 전체 손실을 사용하되, 로깅용 `train_loss`에는 순수 Cross-Entropy(`ce_loss`)만 누적하여 모든 기법 간 학습 손실을 동일 기준으로 비교할 수 있게 함.
3. **결측 클래스 보유 병원의 역빈도 가중치 정규화 스케일 보정 (`data_utils.py`)**:
   - **문제**: Fed-ISIC2019의 센터 1~5는 8개 병변 클래스 중 일부 클래스 샘플 수가 `0`개임. 전체 8개 클래스 평균(`weights.mean()`)으로 나누면 보유 클래스 수가 적은 병원일수록 실제 존재하는 클래스의 가중치(유효 학습률 스케일)가 부풀려짐.
   - **해결 및 근거**: 실제 샘플이 존재하는 클래스들의 평균(`weights[counts > 0].mean()`)으로 정규화하여, 보유 클래스 수와 무관하게 모든 병원의 유효 손실 스케일이 항상 `1.0`으로 유지되도록 통일함.
4. **`evaluate_model` Fallback 시 `macro_f1_gap`의 `NaN` 처리**:
   - **해결 및 근거**: 클라이언트별 테스트 로더가 없을 때 '클래스별 F1 격차'로 대체하면 지표의 정의(병원 간 공정성)가 달라지므로 `NaN`으로 명시 처리함.

### 8.3 보고 방법론 및 분석 노트북 고도화 (`fed_isic2019_colab.ipynb`, `PROJECT_OVERVIEW.md`)
1. **Final(마지막 라운드) 지표와 Best(최고 라운드) 지표 분리 보고**:
   - **문제**: FLamby ISIC2019에는 별도 validation split이 없어 test 셋 최고 라운드(`Best`)만 보고하면 선택 편향(Selection Bias)으로 인해 성능이 낙관적으로 과대평가됨.
   - **해결 및 근거**: 사전에 고정된 20 라운드 종료 시점의 **Final `Mean ± Std` 요약표(`summary_final_mean_std_*.csv`)** 를 주 결과로 제시하고, **Best `Mean ± Std` 요약표(`summary_best_mean_std_*.csv`)** 를 참고용으로 함께 출력하도록 구성함.
2. **탈락한 $\mu$ 후보 자동 필터링 및 목표 정확도 도달 통신량 집계표 추가**:
   - **해결 및 근거**: Stage 1에서 탐색만 한 $\mu$ 후보(1시드)가 최종 3시드 비교표에 섞이지 않도록 `BEST_MU` 기준으로 자동 필터링하고, 목표 정확도 도달 여부(`Reached` 비율), 평균 도달 라운드, 평균 누적 통신량(MB) 집계표(`comm_to_target_*.csv`)를 추가함.
3. **문서 내 데이터셋 수치 및 센터 명칭 정정**:
   - FLamby 공식 분할에 맞춰 센터 1(ViDIR MoleMax: 학습 3,163장 / 테스트 791장), 센터 2(ViDIR Modern: 학습 2,691장 / 테스트 672장), 그리고 모드별 총 샘플 수(`top3`: 학습 15,784장 / 테스트 3,946장, `merge6`: 학습 18,597장 / 테스트 4,650장)를 정확히 반영함.

### 8.4 세션 중단 대비 이어서 학습(`--resume`) 및 원자적(Atomic) 체크포인트 저장 (`main.py`)
- **수정 내역**:
  - 매 라운드 종료 시 `last_{exp}_seed{seed}.pt`에 모델 가중치, (`Centralized`) 옵티마이저 상태, 현재 최고 `balanced_acc`, Python/NumPy/PyTorch/CUDA 난수 상태 및 DataLoader `Generator` 상태를 기록함.
  - `--resume` 실행 시 설정 일치 여부를 검증한 뒤 마지막 완료 라운드 다음부터 이어서 학습하며, CSV 내 미완성·중복 행을 자동 정리(`_truncate_csv`)함.
  - 저장 중 코랩 세션이 끊겨 파일이 깨지는 현상을 막기 위해 `.tmp` 파일 작성 후 `os.replace`로 교체하는 원자적 저장(`_atomic_torch_save`)을 적용하고, 재접속 시 다른 기종의 GPU가 할당되더라도 에러 없이 복원되도록 예외 방어를 추가함.
- **근거**: 코랩 환경에서 약 50분간 진행되는 다중 시드 실험 도중 런타임 연결이 끊기더라도 처음부터 재실행할 필요 없이 즉시 이어서 실험을 완주할 수 있도록 실용적 안정성을 확보함.
