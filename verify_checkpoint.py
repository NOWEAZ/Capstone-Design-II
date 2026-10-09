"""
verify_checkpoint.py  –  2주 차: 체크포인트 무결성 평가
=========================================================
README 제약:
  - 저장 직전 ACC / Macro-F1 값과 로드 후 값이 소수점 끝까지 일치해야 함
  - 모델을 완전히 새로 초기화한 뒤 체크포인트를 로드하여 검증
  - 전처리: Normalize([0.5]*3, [0.5]*3), 증강 없음
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pilot_train import (
    build_custom_resnet18,
    build_loaders,
    evaluate,
    set_seed,
    CHECKPOINT_DIR,
    PILOT_SEED,
)
from metrics import calculate_local_macro_f1, calculate_ece

CKPT_PATH = CHECKPOINT_DIR / "pilot_final.pt"


# ─────────────────────────────────────────────────────────────────────────────
# 1. 체크포인트 저장 (pilot_train.py 가 저장하지 않은 경우의 폴백)
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model: nn.Module,
                    metrics_before_save: dict,
                    path: Path = CKPT_PATH,
                    extra: dict | None = None) -> None:
    """
    모델 state_dict 와 저장 직전 검증 메트릭을 함께 저장한다.

    Parameters
    ----------
    model               : 학습 완료된 모델
    metrics_before_save : {"accuracy": float, "macro_f1": float, "ece": float}
    path                : 저장 경로 (.pt)
    extra               : 추가로 함께 저장할 임의 데이터
    """
    payload: dict = {
        "state_dict":    model.state_dict(),
        "pre_save_mets": metrics_before_save,
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  체크포인트 저장 완료: {path}")
    print(f"  저장 직전 ACC={metrics_before_save['accuracy']:.10f}  "
          f"F1={metrics_before_save['macro_f1']:.10f}  "
          f"ECE={metrics_before_save['ece']:.10f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 무결성 검증 (핵심 로직)
# ─────────────────────────────────────────────────────────────────────────────
def verify_checkpoint_integrity(
    ckpt_path: Path = CKPT_PATH,
    device: torch.device | None = None,
    root: str = "./data",
    tol: float = 0.0,          # 허용 오차 (소수점 끝까지 = 0.0)
) -> bool:
    """
    저장된 체크포인트를 완전히 새로 초기화한 모델에 로드한 뒤,
    전체 Validation 데이터에 대한 ACC / Macro-F1 값이
    저장 직전 수치와 소수점 끝까지 일치하는지 검증한다.

    Parameters
    ----------
    ckpt_path : 검증할 체크포인트 경로
    device    : None 이면 자동 감지
    tol       : 허용 부동소수점 오차. 0.0 → 정확히 일치해야 함.

    Returns
    -------
    bool : 검증 통과 여부
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}\n"
            "먼저 pilot_train.py 를 실행하세요."
        )

    # ── 체크포인트 로드 ───────────────────────────────────────────
    print(f"\n체크포인트 로드: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)

    state_dict      = payload["state_dict"]
    pre_save_mets   = payload["pre_save_mets"]
    saved_res       = payload.get("best_res", 64)   # 해상도 (기본 64)

    expected_acc = pre_save_mets["accuracy"]
    expected_f1  = pre_save_mets["macro_f1"]
    expected_ece = pre_save_mets["ece"]

    print(f"  [저장 직전] ACC={expected_acc:.10f}  "
          f"F1={expected_f1:.10f}  ECE={expected_ece:.10f}")

    # ── 완전히 새로운 모델 초기화 + 가중치 로드 ─────────────────
    print("  새 모델 초기화 → 체크포인트 가중치 로드 중...")
    fresh_model = build_custom_resnet18().to(device)
    fresh_model.load_state_dict(state_dict)
    print("  state_dict 로드 완료.")

    # ── Validation 데이터 통과 (재현성 시드 고정) ────────────────
    set_seed(PILOT_SEED)  # 데이터 로더 셔플 없으므로 사실상 영향 없음
    _, val_loader = build_loaders(image_size=saved_res, root=root)
    loaded_mets   = evaluate(fresh_model, val_loader, device)

    loaded_acc = loaded_mets["accuracy"]
    loaded_f1  = loaded_mets["macro_f1"]
    loaded_ece = loaded_mets["ece"]

    print(f"  [로드 후   ] ACC={loaded_acc:.10f}  "
          f"F1={loaded_f1:.10f}  ECE={loaded_ece:.10f}")

    # ── 수치 비교 ─────────────────────────────────────────────────
    passed = True
    checks = [
        ("Accuracy",   expected_acc, loaded_acc),
        ("Macro-F1",   expected_f1,  loaded_f1),
        ("ECE",        expected_ece, loaded_ece),
    ]

    print("\n  ── 무결성 검증 결과 ─────────────────────────────────")
    for metric_name, expected, actual in checks:
        diff = abs(expected - actual)
        ok   = diff <= tol
        if not ok:
            passed = False
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{status}] {metric_name:<12} "
              f"expected={expected:.15f}  "
              f"actual={actual:.15f}  "
              f"diff={diff:.2e}")

    # ── assert 구문 (소수점 끝까지 일치 강제) ───────────────────
    assert loaded_acc == expected_acc, (
        f"[무결성 실패] Accuracy 불일치: "
        f"expected={expected_acc!r}, got={loaded_acc!r}"
    )
    assert loaded_f1 == expected_f1, (
        f"[무결성 실패] Macro-F1 불일치: "
        f"expected={expected_f1!r}, got={loaded_f1!r}"
    )
    assert loaded_ece == expected_ece, (
        f"[무결성 실패] ECE 불일치: "
        f"expected={expected_ece!r}, got={loaded_ece!r}"
    )

    print(f"\n  {'✓ 모든 지표 일치 – 체크포인트 무결성 검증 통과!' if passed else '✗ 무결성 검증 실패'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# 3. 독립 실행 시 – 최소 학습 후 저장→검증 데모
# ─────────────────────────────────────────────────────────────────────────────
def _demo_save_and_verify(device: torch.device) -> None:
    """
    pilot_train.py 의 결과 체크포인트가 없을 때의 자급 데모.
    2 Epoch 만 학습 후 저장 → 검증.
    """
    from pilot_train import (
        train_model, load_init_weights, _build_optimizer
    )

    print("\n[데모] pilot_final.pt 없음 → 2 Epoch 데모 학습 실행")
    set_seed(PILOT_SEED)

    train_loader, val_loader = build_loaders(image_size=64)
    model = load_init_weights(PILOT_SEED, device)
    opt   = _build_optimizer("sgd", model)

    train_model(model, train_loader, val_loader,
                opt, device, epochs=2, label="demo")

    mets_before = evaluate(model, val_loader, device)
    save_checkpoint(model, mets_before, CKPT_PATH,
                    extra={"best_res": 64, "best_opt": "sgd",
                           "pilot_seed": PILOT_SEED})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("="*60)
    print("체크포인트 무결성 평가")
    print("="*60)

    # pilot_train.py 가 이미 저장한 체크포인트가 없으면 데모 실행
    if not CKPT_PATH.exists():
        _demo_save_and_verify(device)

    passed = verify_checkpoint_integrity(
        ckpt_path=CKPT_PATH,
        device=device,
        tol=0.0,  # 소수점 끝까지 정확히 일치
    )

    if passed:
        print("\n✓ verify_checkpoint.py 완료 – 무결성 검증 통과.")
    else:
        print("\n✗ verify_checkpoint.py 완료 – 무결성 검증 실패!")
        raise SystemExit(1)

