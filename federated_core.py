import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm


def build_efficientnet_b0(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """
    ImageNet 사전학습된 EfficientNet-B0를 불러오고 분류기를 num_classes(기본 8)로 교체합니다.
    """
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        # 구버전 torchvision 호환
        model = models.efficientnet_b0(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_weight_divergence(global_model: nn.Module, local_model: nn.Module) -> float:
    """
    글로벌 모델과 로컬 모델 파라미터 간의 L2 거리(Weight Divergence)를 계산합니다.
    ||w_local - w_global||_2
    """
    sq_sum = 0.0
    global_params = dict(global_model.named_parameters())
    local_params = dict(local_model.named_parameters())

    with torch.no_grad():
        for name, g_param in global_params.items():
            l_param = local_params[name]
            diff = l_param.detach().float() - g_param.detach().float().to(l_param.device)
            sq_sum += torch.sum(diff * diff).item()

    return float(sq_sum ** 0.5)


def compute_model_size_mb(model: nn.Module) -> float:
    """
    모델 state_dict(파라미터 및 버퍼)의 전송 크기를 MB(Megabytes, 1024^2 bytes) 단위로 계산합니다.
    """
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return float(total_bytes / (1024 ** 2))


def compute_round_comm_mb(model: nn.Module, num_clients: int) -> float:
    """
    1 통신 라운드당 총 양방향(Server->Client 다운로드 + Client->Server 업로드) 통신량(MB)을 계산합니다.
    2 * num_clients * model_size_mb
    """
    return float(2.0 * num_clients * compute_model_size_mb(model))


def train_one_client(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    lr: float,
    local_epochs: int,
    device: torch.device | str,
    mu: float = 0.0,
    global_model: nn.Module | None = None,
) -> tuple[nn.Module, float]:
    """
    단일 클라이언트(또는 Centralized) 모델을 학습합니다.
    - mu == 0.0: FedAvg (표준 Cross-Entropy 학습)
    - mu > 0.0: FedProx (Objective에 (mu / 2) * ||w - w_global||_2^2 정규화 항 추가)

    Args:
        model: 학습할 로컬 모델
        dataloader: 클라이언트 학습 DataLoader
        criterion: 손실 함수 (예: 가중 CrossEntropyLoss)
        lr: 학습률
        local_epochs: 로컬 에폭 수
        device: 학습 디바이스
        mu: FedProx 근접 항 계수 (0.0이면 FedAvg)
        global_model: FedProx 기준이 되는 글로벌 모델 (None이면 학습 시작 시점의 model 스냅샷 사용)

    Returns:
        tuple: (학습된 local_model, 평균 train_loss)
    """
    model.to(device)
    model.train()

    if mu > 0.0:
        if global_model is None:
            global_params = [p.detach().clone().to(device) for p in model.parameters()]
        else:
            global_params = [p.detach().clone().to(device) for p in global_model.parameters()]
    else:
        global_params = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for images, targets in tqdm(dataloader, desc="Local Train", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long().view(-1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)

            if mu > 0.0 and global_params is not None:
                prox_term = torch.tensor(0.0, device=device)
                for local_p, global_p in zip(model.parameters(), global_params):
                    prox_term = prox_term + torch.sum((local_p - global_p) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    return model, avg_loss


def server_aggregate(
    global_model: nn.Module,
    client_models: list[nn.Module],
    client_sample_counts: list[int],
) -> nn.Module:
    """
    클라이언트들의 샘플 수에 비례한 가중 평균으로 글로벌 모델 파라미터를 집계(Aggregation)합니다.

    Args:
        global_model: 업데이트할 글로벌 모델
        client_models: 학습 완료된 로컬 모델 리스트
        client_sample_counts: 각 클라이언트의 데이터 샘플 수 리스트

    Returns:
        nn.Module: 파라미터가 업데이트된 글로벌 모델
    """
    if len(client_models) == 0:
        raise ValueError("client_models 리스트가 비어 있습니다.")

    total_samples = float(sum(client_sample_counts))
    if total_samples <= 0:
        raise ValueError("전체 클라이언트 샘플 수 합은 0보다 커야 합니다.")

    weights = [count / total_samples for count in client_sample_counts]
    global_state = copy.deepcopy(global_model.state_dict())
    client_states = [m.state_dict() for m in client_models]

    with torch.no_grad():
        for key in global_state.keys():
            target_tensor = global_state[key]
            # 실수형(float) 파라미터/버퍼는 가중 평균, 정수형 버퍼(num_batches_tracked 등)는 가중 평균 후 반올림
            if torch.is_floating_point(target_tensor):
                accum = torch.zeros_like(target_tensor, dtype=torch.float32)
                for w, c_state in zip(weights, client_states):
                    accum += w * c_state[key].to(accum.device, dtype=torch.float32)
                global_state[key] = accum.to(dtype=target_tensor.dtype)
            else:
                accum = torch.zeros_like(target_tensor, dtype=torch.float32)
                for w, c_state in zip(weights, client_states):
                    accum += w * c_state[key].to(accum.device, dtype=torch.float32)
                global_state[key] = torch.round(accum).to(dtype=target_tensor.dtype)

    global_model.load_state_dict(global_state)
    return global_model


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)를 계산합니다.
    예측 확률(Confidence)과 실제 정확도(Accuracy) 간의 가중 평균 오차를 측정합니다.

    Args:
        probs: shape (N, C)의 소프트맥스 예측 확률 배열
        targets: shape (N,)의 정답 레이블 배열
        n_bins: 신뢰도 구간(bin) 개수 (기본값: 15)

    Returns:
        float: ECE 값 (0에 가까울수록 보정이 잘 된 모델)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(targets)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device | str,
    criterion: nn.Module,
    desc: str = "Evaluating",
):
    """데이터로더에 대해 손실, 예측 확률, 예측 클래스, 정답 레이블을 수집합니다."""
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long().view(-1)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / max(total_samples, 1)

    return avg_loss, y_prob, y_pred, y_true


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device | str,
    criterion: nn.Module | None = None,
    client_test_loaders: list | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """
    글로벌 테스트 DataLoader에서 모델을 평가하여
    Balanced Accuracy, Macro F1-Score, ECE(Expected Calibration Error),
    그리고 클라이언트별 최고-최저 Macro-F1 격차(Fairness 지표)를 계산합니다.

    Returns:
        dict: {
            "loss": float,
            "balanced_acc": float,
            "macro_f1": float,
            "ece": float,
            "max_macro_f1": float,
            "min_macro_f1": float,
            "macro_f1_gap": float,
        }
    """
    model.to(device)
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    avg_loss, y_prob, y_pred, y_true = _collect_predictions(
        model, dataloader, device, criterion, desc="Evaluating Global"
    )

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ece = compute_ece(y_prob, y_true, n_bins=n_bins)

    # 클라이언트별 테스트 로더가 있으면 클라이언트 간 최고-최저 Macro-F1 계산
    if client_test_loaders is None and hasattr(dataloader, "client_test_loaders"):
        client_test_loaders = dataloader.client_test_loaders

    if client_test_loaders:
        client_f1s = []
        for idx, c_loader in enumerate(client_test_loaders):
            _, _, c_pred, c_true = _collect_predictions(
                model, c_loader, device, criterion, desc=f"Eval Client {idx}"
            )
            c_f1 = float(f1_score(c_true, c_pred, average="macro", zero_division=0))
            client_f1s.append(c_f1)
        max_macro_f1 = float(max(client_f1s))
        min_macro_f1 = float(min(client_f1s))
        macro_f1_gap = float(max_macro_f1 - min_macro_f1)
    else:
        # 클라이언트별 테스트 로더가 없을 경우 클래스별 F1의 최고-최저 차이로 대체
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        max_macro_f1 = float(np.max(per_class_f1))
        min_macro_f1 = float(np.min(per_class_f1))
        macro_f1_gap = float(max_macro_f1 - min_macro_f1)

    return {
        "loss": avg_loss,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "ece": ece,
        "max_macro_f1": max_macro_f1,
        "min_macro_f1": min_macro_f1,
        "macro_f1_gap": macro_f1_gap,
    }

