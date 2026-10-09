"""
pilot_train.py  –  2주 차: 중앙집중형 파일럿 테스트
=====================================================
README 제약:
  - BatchNorm → GroupNorm(num_groups=2) 완전 교체
  - 첫 Conv: kernel=3, stride=1, padding=1 / MaxPool 제거
  - 전역 set_seed() 함수로 재현성 보장 (cudnn.deterministic=True)
  - 시드별 초기 state_dict 저장/로드 헬퍼
  - 전처리: Normalize([0.5]*3, [0.5]*3), 증강 없음
  - 탐색 시드: 41 고정
  - 파일럿 세부: 해상도 비교 → Optimizer 비교 → 라운드(Epoch) 비교
"""

from __future__ import annotations

import os
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.resnet import BasicBlock
import torchvision.transforms as transforms
from medmnist import BloodMNIST

from metrics import calculate_local_macro_f1, calculate_ece

# ── 상수 ─────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 8
BATCH_SIZE    = 128
PILOT_SEED    = 41
CHECKPOINT_DIR = Path("outputs/pilot")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

INIT_WEIGHTS_DIR = Path("outputs/init_weights")
INIT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 전역 시드 제어 (README §2.4)
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Python / NumPy / PyTorch (CPU+CUDA) 모든 난수 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. CustomResNet18 – BN→GN, Conv1 수정, MaxPool 제거 (README §2.1~2.2)
# ─────────────────────────────────────────────────────────────────────────────
def _replace_bn_with_gn(module: nn.Module, num_groups: int = 2) -> nn.Module:
    """재귀적으로 BatchNorm2d → GroupNorm(num_groups) 교체."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=num_groups,
                              num_channels=child.num_features)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


def build_custom_resnet18(num_classes: int = NUM_CLASSES,
                           num_groups: int = 2) -> nn.Module:
    """
    torchvision ResNet-18 을 기반으로:
      1) BatchNorm2d → GroupNorm(num_groups=2) 전면 교체
      2) conv1: 7×7/stride2 → 3×3/stride1/padding1
      3) maxpool 제거
      4) fc: 1000 → num_classes
    """
    model = models.resnet18(weights=None)

    # ── Step 1: BN → GN ──────────────────────────────────────────
    _replace_bn_with_gn(model, num_groups=num_groups)

    # ── Step 2: 첫 번째 Conv 교체 ────────────────────────────────
    original_conv1 = model.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    model.conv1 = nn.Conv2d(
        in_channels  = original_conv1.in_channels,
        out_channels = original_conv1.out_channels,
        kernel_size  = 3,
        stride       = 1,
        padding      = 1,
        bias         = False,
    )

    # ── Step 3: MaxPool 제거 → Identity ──────────────────────────
    model.maxpool = nn.Identity()

    # ── Step 4: 출력층 교체 ───────────────────────────────────────
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. 초기 가중치 저장/로드 헬퍼 (README §2.3)
# ─────────────────────────────────────────────────────────────────────────────
def save_init_weights(seed: int,
                      save_dir: Path = INIT_WEIGHTS_DIR) -> Path:
    """
    seed 에 맞게 모델을 초기화한 뒤 state_dict 를 파일로 저장한다.
    동일한 파일이 이미 있으면 덮어씌우지 않고 그대로 반환한다.
    """
    path = save_dir / f"init_seed{seed}.pt"
    if path.exists():
        print(f"  초기 가중치 파일 이미 존재 (skip): {path}")
        return path

    set_seed(seed)
    model = build_custom_resnet18()
    torch.save(model.state_dict(), path)
    print(f"  초기 가중치 저장: {path}")
    return path


def load_init_weights(seed: int,
                      device: torch.device,
                      save_dir: Path = INIT_WEIGHTS_DIR) -> nn.Module:
    """
    저장된 초기 state_dict 를 로드하여 새 모델에 적용 후 반환한다.
    파일이 없으면 먼저 생성한다.
    """
    path = save_dir / f"init_seed{seed}.pt"
    if not path.exists():
        save_init_weights(seed, save_dir)

    model = build_custom_resnet18().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"  초기 가중치 로드: {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. 데이터 로더 (README §3.4: Normalize 고정, 증강 없음)
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(image_size: int) -> transforms.Compose:
    """image_size 로 리사이즈 후 정규화. 증강 없음."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_loaders(image_size: int,
                  batch_size: int = BATCH_SIZE,
                  root: str = "./data") -> tuple[DataLoader, DataLoader]:
    """Train / Val DataLoader 를 반환한다."""
    tf_train = get_transform(image_size)
    tf_val   = get_transform(image_size)

    train_ds = BloodMNIST(split="train", transform=tf_train,
                           download=True, root=root)
    val_ds   = BloodMNIST(split="val",   transform=tf_val,
                           download=True, root=root)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 5. 학습 / 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """1 Epoch 학습 후 평균 Loss 반환."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).squeeze(1).long()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> dict:
    """
    전체 Validation 셋에 대해 Accuracy, Local Macro-F1, ECE 를 계산한다.
    Returns dict with keys: accuracy, macro_f1, ece
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu().squeeze(1))

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)

    probs  = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    preds  = probs.argmax(axis=1)

    # Accuracy
    acc = float((preds == labels_np).mean())

    # Local Macro-F1 (검증셋에 실존하는 클래스만)
    present = sorted(np.unique(labels_np).tolist())
    f1  = calculate_local_macro_f1(labels_np, preds, present)

    # ECE
    ece = calculate_ece(labels_np, probs)

    return {"accuracy": acc, "macro_f1": f1, "ece": ece}


# ─────────────────────────────────────────────────────────────────────────────
# 6. 학습 실행 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    label: str = "",
    record_epochs: list[int] | None = None,
) -> tuple[list[dict], dict[int, dict]]:
    """
    지정 Epoch 수만큼 학습하며 매 Epoch 결과를 반환한다.

    Parameters
    ----------
    record_epochs : 체크포인트 기록이 필요한 특정 Epoch 번호 (1-indexed)

    Returns
    -------
    history        : 매 Epoch dict (epoch, loss, accuracy, macro_f1, ece)
    epoch_records  : {epoch_num: metrics_dict}  (record_epochs 지정 시)
    """
    criterion    = nn.CrossEntropyLoss()
    history: list[dict] = []
    epoch_records: dict[int, dict] = {}
    record_set   = set(record_epochs or [])

    for ep in range(1, epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        mets = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {"epoch": ep, "loss": loss, **mets}
        history.append(row)

        if ep in record_set:
            epoch_records[ep] = dict(mets)

        status = (f"[{label}] Ep {ep:>3}/{epochs}  "
                  f"loss={loss:.4f}  acc={mets['accuracy']:.4f}  "
                  f"F1={mets['macro_f1']:.4f}  ECE={mets['ece']:.4f}  "
                  f"({elapsed:.1f}s)")
        print(status)

    return history, epoch_records


def _build_optimizer(name: str,
                     model: nn.Module,
                     lr: float = 1e-3) -> optim.Optimizer:
    """Optimizer 이름 문자열로 인스턴스 생성."""
    name = name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                         weight_decay=1e-4)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"지원하지 않는 optimizer: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 파일럿 실행 (탐색 시드 41 고정)
# ─────────────────────────────────────────────────────────────────────────────
def pilot_resolution_comparison(device: torch.device) -> int:
    """
    Pilot Step 1: 28×28 vs 64×64, SGD, 10 Epoch
    Returns: 승리 해상도 (28 or 64)
    """
    print("\n" + "="*60)
    print("Pilot Step 1: 해상도 비교 (28x28 vs 64x64, SGD, 10 Epoch)")
    print("="*60)

    results: dict[int, float] = {}
    for res in [28, 64]:
        train_loader, val_loader = build_loaders(image_size=res)

        model = load_init_weights(PILOT_SEED, device)
        opt   = _build_optimizer("sgd", model)

        label   = f"SGD res={res}"
        history, _ = train_model(model, train_loader, val_loader,
                                  opt, device, epochs=10, label=label)

        best_f1 = max(row["macro_f1"] for row in history)
        final   = history[-1]["macro_f1"]
        results[res] = final
        print(f"  → 해상도 {res}x{res}  Final Macro-F1={final:.4f}  "
              f"Best={best_f1:.4f}")

    winner = max(results, key=results.get)  # type: ignore[arg-type]
    print(f"\n  ✓ 해상도 비교 승리: {winner}x{winner} "
          f"(F1={results[winner]:.4f})")
    return winner


def pilot_optimizer_comparison(device: torch.device,
                                best_res: int) -> str:
    """
    Pilot Step 2: SGD vs Adam, 승리 해상도 적용, 10 Epoch
    Returns: 승리 optimizer 이름 ("sgd" or "adam")
    """
    print("\n" + "="*60)
    print(f"Pilot Step 2: Optimizer 비교 (SGD vs Adam, res={best_res}, 10 Epoch)")
    print("="*60)

    train_loader, val_loader = build_loaders(image_size=best_res)

    results: dict[str, float] = {}
    for opt_name in ["sgd", "adam"]:
        model = load_init_weights(PILOT_SEED, device)
        opt   = _build_optimizer(opt_name, model)

        label   = f"{opt_name.upper()} res={best_res}"
        history, _ = train_model(model, train_loader, val_loader,
                                  opt, device, epochs=10, label=label)

        final = history[-1]["macro_f1"]
        results[opt_name] = final
        print(f"  → {opt_name.upper()}  Final Macro-F1={final:.4f}")

    winner = max(results, key=results.get)  # type: ignore[arg-type]
    print(f"\n  ✓ Optimizer 비교 승리: {winner.upper()} "
          f"(F1={results[winner]:.4f})")
    return winner


def pilot_epoch_comparison(device: torch.device,
                            best_res: int,
                            best_opt: str) -> dict:
    """
    Pilot Step 3: 최종 설정으로 40 Epoch 학습.
      - 30 Epoch 시점 성능 기록
      - 31~40 Epoch 구간 최고 성능 개선 폭 기록
    Returns: 결과 dict
    """
    print("\n" + "="*60)
    print(f"Pilot Step 3: 라운드 비교 (res={best_res}, {best_opt.upper()}, 40 Epoch)")
    print("="*60)

    train_loader, val_loader = build_loaders(image_size=best_res)
    model = load_init_weights(PILOT_SEED, device)
    opt   = _build_optimizer(best_opt, model)

    label = f"{best_opt.upper()} res={best_res} 40ep"
    history, epoch_records = train_model(
        model, train_loader, val_loader,
        opt, device, epochs=40,
        label=label,
        record_epochs=[30],
    )

    ep30_mets = epoch_records[30]
    ep30_f1   = ep30_mets["macro_f1"]

    # 31~40 구간 최고 성능 개선 폭
    f1_31_40  = [history[ep - 1]["macro_f1"] for ep in range(31, 41)]
    best_f1_late = max(f1_31_40)
    improvement  = best_f1_late - ep30_f1

    print(f"\n  ─ 30 Epoch 시점: "
          f"Macro-F1={ep30_f1:.4f}  "
          f"ACC={ep30_mets['accuracy']:.4f}  "
          f"ECE={ep30_mets['ece']:.4f}")
    print(f"  ─ 31~40 구간 최고 Macro-F1: {best_f1_late:.4f}")
    print(f"  ─ 최고 성능 개선 폭 (31~40 - 30ep): {improvement:+.4f}")

    result = {
        "ep30":         ep30_mets,
        "best_f1_31_40": best_f1_late,
        "improvement":  improvement,
        "history":      history,
        "model":        model,           # 최종 학습 완료 모델
        "val_loader":   val_loader,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(PILOT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 탐색 시드 41 초기 가중치 사전 저장 (최초 1회)
    save_init_weights(PILOT_SEED)

    # ── Step 1: 해상도 비교 ──────────────────────────────────────
    best_res = pilot_resolution_comparison(device)

    # ── Step 2: Optimizer 비교 ──────────────────────────────────
    best_opt = pilot_optimizer_comparison(device, best_res)

    # ── Step 3: 라운드 비교 ─────────────────────────────────────
    pilot_result = pilot_epoch_comparison(device, best_res, best_opt)

    # ── 최종 모델 체크포인트 저장 (verify_checkpoint.py 에서 사용) ─
    ckpt_path = CHECKPOINT_DIR / "pilot_final.pt"
    final_model = pilot_result["model"]
    val_loader  = pilot_result["val_loader"]

    # 저장 직전 메트릭 계산
    pre_save_mets = evaluate(final_model, val_loader, device)
    torch.save({
        "state_dict":     final_model.state_dict(),
        "pre_save_mets":  pre_save_mets,
        "best_res":       best_res,
        "best_opt":       best_opt,
        "pilot_seed":     PILOT_SEED,
    }, ckpt_path)

    print(f"\n  체크포인트 저장 완료: {ckpt_path}")
    print(f"  저장 직전 ACC={pre_save_mets['accuracy']:.6f}  "
          f"F1={pre_save_mets['macro_f1']:.6f}  "
          f"ECE={pre_save_mets['ece']:.6f}")
    print("\n✓ pilot_train.py 완료.")

