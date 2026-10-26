# 🏥 Fed-ISIC2019 의료 영상 연합학습 프로젝트 개요 (Project Overview)

## 1. 프로젝트 목적
본 프로젝트는 다기관 피부병변 분류 데이터셋인 **FLamby Fed-ISIC2019 (8-Class)** 환경에서 발생하는 **병원 간 데이터 비독립·비동등 분포(Non-IID: Feature Shift & Label Imbalance)** 문제를 분석하고, 표준 연합학습 알고리즘인 **FedAvg**와 근접 정규화 기반 **FedProx**, 그리고 상한 기준선인 **Centralized** 학습의 성능을 다각도로 비교·검증하기 위해 설계되었습니다.

단순 분류 정확도 비교에 그치지 않고, 실제 의료 AI 임상 적용에서 필수적인 **신뢰도 보정(ECE)**, **병원 간 공정성(최고-최저 Macro-F1 격차)**, **가중치 발산도(Weight Divergence)**, 그리고 **통신 효율성(Communication-to-Accuracy)** 을 종합적으로 평가합니다.

---

## 2. 워크스페이스 파일 구조

| 파일명 | 역할 및 주요 기능 |
| :--- | :--- |
| **`requirements.txt`** | 필수 파이썬 패키지 목록 (`torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`) |
| **`data_utils.py`** | `FedIsic2019` 데이터셋 기반 `top3` / `merge6` 분할 로더(`get_dataloaders`), 시드 고정 워커, 역빈도 클래스 가중치 계산(`compute_class_weights`) |
| **`federated_core.py`** | `EfficientNet-B0` 모델 생성, `FedAvg`/`FedProx` 로컬 학습(`train_one_client`), 서버 가중 평균 집계(`server_aggregate`), 종합 평가지표(`evaluate_model`, `compute_ece`), 가중치 발산도 및 통신량 계산 |
| **`main.py`** | CLI 인자(`argparse`) 파싱, 난수 시드 고정(`set_seed`), 라운드별 학습/평가 루프 실행, 실시간 CSV 로깅 및 최고 `balanced_acc` 모델 체크포인트(`.pt`) 저장 |
| **`fed_isic2019_colab.ipynb`** | 팀원들이 Google Colab에서 Phase별(Stage 1 단일 시드 탐색 $\rightarrow$ Stage 2 3시드 검증 $\rightarrow$ 통합 요약표 및 그래프 생성)로 실행할 수 있는 주피터 노트북 |
| **`DESIGN_DECISIONS_AND_CHANGELOG.md`** | 프로젝트 진행 과정에서 결정된 아키텍처·평가지표·하이퍼파라미터 수정 내역과 기술적 근거 문서 |

---

## 3. 데이터셋 분할 모드 (`--mode`)

FLamby의 `FedIsic2019`는 총 6개 병원/센터(Center 0 ~ 5)로 구성되어 있으며, 본 프로젝트에서는 3개 클라이언트 환경에서 두 가지 분할 시나리오를 비교합니다.

1. **`mode = 'top3'` (상위 3개 주요 센터)**
   - **Client 0**: Center 0 (BCN, 약 9,930장)
   - **Client 1**: Center 1 (HAM10000 - ViDIR Modern, 약 3,163장)
   - **Client 2**: Center 2 (HAM10000 - ViDIR MoleMax, 약 3,160장)
2. **`mode = 'merge6'` (전체 6개 센터를 3개 클라이언트로 병합)**
   - 대형 센터(0, 1, 2) 각각에 소형 센터(3, 4, 5)를 페어링하여 모든 6개 병원의 도메인 특성을 포함하도록 구성:
   - **Client 0**: Center 0 + Center 3
   - **Client 1**: Center 1 + Center 4
   - **Client 2**: Center 2 + Center 5

---

## 4. 모델 아키텍처 및 학습 기법 (`--method`)

### 4.1 모델 및 손실 함수
- **백본 모델**: ImageNet 사전학습된 `EfficientNet-B0` (`BatchNorm2d` 유지, 최종 분류기 `classifier[1]`을 8클래스 `nn.Linear(1280, 8)`로 교체)
- **클래스 불균형 해소**: 각 클라이언트의 로컬 클래스 빈도에 반비례하는 **역빈도 가중치(Inverse-Frequency Class Weights)** 를 적용한 `nn.CrossEntropyLoss(weight=...)` 사용
  $$w_c = \frac{N}{K \cdot n_c} \quad \text{(평균이 1이 되도록 정규화)}$$

### 4.2 비교 학습 기법
1. **`centralized`**: 3개 클라이언트의 학습 데이터를 하나로 통합하여 학습하는 상한선(Upper-bound) 기준 모델 (`local_epochs = 1`, `rounds = 20`)
2. **`fedavg` ($\mu = 0.0$)**: 각 클라이언트가 로컬 데이터로 `local_epochs = 2` 동안 독립 학습한 뒤, 서버에서 샘플 수에 비례한 가중 평균을 수행하는 표준 연합학습
3. **`fedprox` ($\mu > 0.0$)**: 로컬 목적함수에 글로벌 모델 $w_{\text{global}}$로부터의 이탈을 억제하는 근접 항(Proximal Term)을 추가하여 Non-IID 환경의 **Client Drift**를 완화하는 기법
   $$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{CE}}(w) + \frac{\mu}{2} \| w - w_{\text{global}} \|_2^2$$

---

## 5. 핵심 평가지표 (Evaluation Metrics)

매 통신 라운드 종료 시 Global Test 셋 및 각 클라이언트별 Test 셋에서 다음 6가지 지표를 측정하여 실시간 CSV로 기록합니다.

| 컬럼명 | 지표 설명 | 방향 |
| :--- | :--- | :---: |
| **`balanced_acc`** | 클래스별 재현율(Recall)의 산술 평균. 극심한 클래스 불균형 환경에서 전체 분류 성능을 대표하며 **최고 모델(`.pt`) 저장 기준**으로 사용 | $\uparrow$ |
| **`macro_f1`** | 8개 클래스 F1-Score의 비가중 평균 | $\uparrow$ |
| **`macro_f1_gap`** | 3개 클라이언트 각각의 로컬 테스트셋에서 측정한 `Macro-F1`의 **최고값 - 최저값 차이** ($\max_k \text{F1}_k - \min_k \text{F1}_k$). **병원 간 공정성(Client Fairness)** 지표 | $\downarrow$ |
| **`ece`** | 15개 구간(`n_bins=15`) 기준 **Expected Calibration Error**. 모델의 예측 확률(Confidence)과 실제 정답률(Accuracy) 간의 오차를 측정하는 **의료 AI 신뢰도 보정** 지표 | $\downarrow$ |
| **`weight_divergence`** | 글로벌 모델과 학습 완료된 로컬 모델 파라미터 간의 평균 L2 거리 ($\|w_{\text{local}} - w_{\text{global}}\|_2$). **Client Drift** 정도를 정량화 | 분석용 |
| **`cumulative_comm_mb`** | 해당 라운드까지 서버-클라이언트 간 송수신된 누적 모델 가중치 통신량(MB). **목표 정확도 도달 통신 효율성(Communication-to-Accuracy)** 평가에 사용 | $\downarrow$ |

---

## 6. 실행 방법 요약

### 6.1 CLI 단독 실행 예시
```bash
# 1. Centralized Baseline (20 라운드, 라운드당 1 에폭)
python main.py --method centralized --mode top3 --rounds 20 --local_epochs 1 --batch_size 32 --lr 1e-4 --seed 42 --save_dir ./results

# 2. FedAvg (20 라운드, 라운드당 로컬 2 에폭)
python main.py --method fedavg --mode top3 --rounds 20 --local_epochs 2 --batch_size 32 --lr 1e-4 --mu 0.0 --seed 42 --save_dir ./results

# 3. FedProx (mu=0.01, 20 라운드, 라운드당 로컬 2 에폭)
python main.py --method fedprox --mode top3 --rounds 20 --local_epochs 2 --batch_size 32 --lr 1e-4 --mu 0.01 --seed 42 --save_dir ./results
```

### 6.2 Google Colab 협업 실행
- **`fed_isic2019_colab.ipynb`** 파일을 코랩에서 열고 **Phase 0 ~ Phase 5** 셀을 순차적으로 실행합니다.
- **Stage 1 (`seed=42`)** 에서 최적의 $\mu$를 먼저 탐색한 뒤, **Stage 2 (`seed=100, 2026`)** 에서 나머지 2개 시드만 추가 실행하면 **Phase 5**에서 3시드 `Mean ± Std` 요약 테이블과 표준편차 음영 그래프가 자동 생성됩니다.

