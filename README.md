# Project README for AI Agent Context & Guidelines

이 문서는 AI 어시스턴트(VSCode Extension)가 본 연합학습(Federated Learning) 시뮬레이션 프로젝트의 도메인 지식, 아키텍처 제약, 데이터 파이프라인 규칙 및 평가 기준을 정확히 이해하고 코드를 생성하도록 작성된 시스템 문서입니다.

AI 에이전트는 코드 작성, 리팩토링, 디버깅 시 아래의 모든 제약 사항을 **반드시 최우선으로 준수**해야 합니다.

---

## 1. Project Overview & Goal
* **프로젝트명**: BloodMNIST 의료 이미지 분류를 위한 연합학습 시스템 구현 및 성능 분석
* **기반 프레임워크**: Python, PyTorch, Flower (`flwr`)
* **핵심 목적**: 단일 PC 환경에서 가상 클라이언트 3개를 시뮬레이션하여 중앙집중형(Centralized), FedAvg, FedProx 세 가지 학습 방법론의 정확도, 최저 성능, 신뢰도(ECE), 통신 효율을 정량 비교
* **대상 데이터셋**: BloodMNIST (8개 클래스, 단일 라벨 다중 분류)

---

## 2. Model Architecture Strict Rules (PyTorch)
PyTorch 모델 설계 시 기본 ResNet-18 구조를 그대로 사용하지 않고 다음 수정 사항을 엄격히 적용해야 합니다.

1. **Normalization Layer 제약**:
   * `nn.BatchNorm2d` 사용을 완전히 금지합니다.
   * 라벨 불균형(non-IID) 환경에서의 배치 통계량(Mean/Variance) 붕괴를 방지하기 위해 **Group Norm (2그룹)**으로 대체하여 구현하세요. (`nn.GroupNorm(num_groups=2, num_channels=...)`)
2. **Input Layer 제약**:
   * 소형 이미지 입력(28x28 또는 64x64)의 공간 정보 손실을 막기 위해 첫 번째 Conv 계층을 `kernel_size=3, stride=1, padding=1`로 변경하세요.
   * 이어지는 `nn.MaxPool2d` 계층은 제거하세요.
3. **가중치 초기화 고정**:
   * 방법론 간 공정한 비교를 위해 시드별(42, 43, 44) 초기 가중치 `state_dict`를 사전에 파일로 저장하고, 중앙집중형/FedAvg/FedProx가 모두 동일한 파일에서 가중치를 로드하도록 작성하세요.
4. **재현성(Reproducibility) 보장**
   * Python random, numpy, torch (CPU 및 CUDA)의 모든 난수 생성기 시드를 고정하는 전역 set_seed(seed) 함수를 반드시 구현하여 코드 최상단 진입점에 배치하세요.
   * torch.backends.cudnn.deterministic = True 설정을 포함하세요.

---

## 3. Data Pipeline & Partitioning Rules
데이터 전처리 및 클라이언트 분할 모듈 작성 시 아래 로직을 엄격히 따라야 합니다.

1. **분할 방식**:
   * **IID**: 클래스별 층화 균등 배분 (Stratified Uniform Allocation)
   * **non-IID**: Dirichlet 분포 ($\alpha=0.5, 0.1$) 기반 비율 추출 후 largest-remainder 정수 배분 방식 적용. 남는 표본은 시드 순서로 순차 배분합니다.
2. **최소 표본 보장 루프**:
   * 추첨 후 단 한 명의 클라이언트라도 학습 데이터 수가 **1,000장 미만**이면 결과를 버리고 1,000장 이상이 나올 때까지 최대 100회까지 재추첨하는 `while` 루프를 구현하세요.
3. **Test 데이터 통제 (Data Leakage 방지)**:
   * 로컬 클라이언트 평가용으로는 `validation` 분할만 제공하며, 클라이언트용 로컬 `test` 셋은 생성하지 마세요.
   * 공식 전체 `test` 데이터셋은 서버 측에 통제해 두고, `eval.test_frozen=true` 플래그 및 접근 사유가 입력된 경우에만 접근을 허용하는 플래그 제어 코드를 작성하세요.
4. **데이터 전처리 제약**
   * 이미지는 [0,1] 범위 변환 후 평균 0.5, 표준편차 0.5로 고정하여 정규화하세요.
   * RandomFlip, RandomCrop 등의 데이터 증강(Augmentation)이나 클래스 불균형 해소를 위한 오버샘플링 로직은 절대 추가하지 마세요.
---

## 4. Federated Learning Pipeline (Flower)
연합학습 파이프라인 구축 시 아래 규칙을 준수하세요.

1. **클라이언트 상태 제약**:
   * 매 라운드 3개 클라이언트가 100% 전원 참여합니다.
   * 클라이언트의 Optimizer(SGD/Adam) 상태는 라운드 간에 유지하지 않고 매 라운드 새로 초기화해야 합니다.
2. **FedProx 구현**:
   * 로컬 학습 목적함수(Loss)에 전역 가중치와의 차이를 제약하는 Proximal Term ($\frac{\mu}{2} \cdot \Vert{}w - w_{global}\Vert{}^2$)을 추가하세요.
   * $\mu$ 후보는 `[0.001, 0.01, 0.1, 1.0]` 중에서 지정된 파일럿 규칙에 따라 단일 값을 정해 적용합니다.
3. **통신량 로깅**:
   * 매 라운드 서버와 클라이언트가 주고받는 모델 가중치 딕셔너리의 실제 payload 바이트(Byte) 용량을 측정한 뒤, 로그 파일에 자동으로 로깅하는 스크립트를 작성하세요.

---

## 5. Evaluation Metrics Implementation
평가 모듈 작성 시 일반적인 단순 Accuracy 외에 아래 커스텀 지표 계산 알고리즘을 정확히 구현하세요.

1. **Local Macro-F1 (주 지표)**:
   * 고정된 8개 클래스 전체로 평균을 내지 마세요.
   * 극단적 non-IID 조건으로 인해 클래스가 부재할 수 있으므로, 해당 클라이언트 검증셋에 **실제로 존재하는(Support > 0) 클래스들의 F1-Score만 평균** 내어 산출하세요.
2. **최고-최저 격차 (Gap)**:
   * 매 평가 시점마다 3개 클라이언트의 Local Macro-F1 중 최댓값과 최솟값의 차이 (`Max - Min`)를 함께 반환하세요.
3. **ECE (Expected Calibration Error)**:
   * 최고 확률(Top-1 Confidence)을 기준으로 예측 확률을 10개의 동일 폭(Equal-width) 구간으로 나눕니다.
   * 빈 구간의 기여도는 0으로 처리하며, 각 구간 내의 `|평균 정확도 - 평균 확신도|`를 해당 구간 표본 비율로 가중 평균하는 함수를 작성하세요.