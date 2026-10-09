"""
metrics.py  –  2주 차: 평가지표 구현 및 단위 테스트
=====================================================
README 제약:
  - Local Macro-F1: 해당 클라이언트 검증셋에 실제로 존재하는 클래스만 F1 평균
  - ECE: Top-1 Confidence 기준 10개 Equal-width 구간, 빈 구간 기여 0
  - 최고-최저 Gap: 3개 클라이언트 Local Macro-F1 의 Max - Min
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Local Macro-F1
# ─────────────────────────────────────────────────────────────────────────────
def calculate_local_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    present_classes: list[int] | np.ndarray,
) -> float:
    """
    해당 클라이언트 검증셋에 실제로 존재하는(support > 0) 클래스들만의
    F1-Score 를 개별 계산한 뒤 단순 평균을 반환한다 (Macro 방식).

    Parameters
    ----------
    y_true          : 정수 레이블 배열, shape (N,)
    y_pred          : 정수 예측 배열, shape (N,)
    present_classes : 해당 검증셋에 존재하는 클래스 인덱스 목록
                      (support > 0 클래스, 외부에서 사전 필터링 후 전달)

    Returns
    -------
    float : present_classes 에 한정한 Macro-F1 (0 ~ 1)

    Notes
    -----
    - present_classes 가 비어 있으면 0.0 을 반환한다.
    - TP/FP/FN 이 모두 0인 예외 클래스는 F1 = 0으로 처리한다.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    present_classes = list(present_classes)

    if not present_classes:
        return 0.0

    f1_scores: list[float] = []
    for c in present_classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ─────────────────────────────────────────────────────────────────────────────
# 2. ECE (Expected Calibration Error)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Top-1 Confidence(최고 예측 확률) 기준 Equal-width 구간으로 ECE를 계산한다.

    Parameters
    ----------
    y_true   : 정수 레이블 배열, shape (N,)
    y_prob   : 클래스별 소프트맥스 확률, shape (N, C)
    num_bins : 구간 수 (기본값 10)

    Returns
    -------
    float : ECE 값 (0 ~ 1). 낮을수록 잘 보정된 모델.

    Algorithm
    ---------
    1. confidence = max(y_prob, axis=1)   (Top-1 확률)
    2. y_hat      = argmax(y_prob, axis=1) (Top-1 예측)
    3. bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]  (마지막 구간은 닫힘)
    4. 각 구간 b:
         acc_b  = mean(y_hat[b] == y_true[b])
         conf_b = mean(confidence[b])
         n_b    = |b|
       빈 구간(n_b = 0) → 기여도 0
    5. ECE = sum_b  (n_b / N) * |acc_b - conf_b|
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2-D (N, C), got shape {y_prob.shape}")

    n = len(y_true)
    if n == 0:
        return 0.0

    confidence = y_prob.max(axis=1)        # (N,)
    y_hat      = y_prob.argmax(axis=1)     # (N,)
    correct    = (y_hat == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # 마지막 구간: 오른쪽 끝(1.0) 포함
        if i < num_bins - 1:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)

        n_b = mask.sum()
        if n_b == 0:
            continue  # 빈 구간 기여도 0

        acc_b  = correct[mask].mean()
        conf_b = confidence[mask].mean()
        ece   += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 최고-최저 Gap (3 클라이언트용)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_f1_gap(client_f1_list: list[float]) -> float:
    """
    3개 클라이언트의 Local Macro-F1 목록에서 Max - Min 격차를 반환한다.

    Parameters
    ----------
    client_f1_list : 각 클라이언트의 Local Macro-F1 값 목록

    Returns
    -------
    float : Max - Min (≥ 0)
    """
    if len(client_f1_list) < 2:
        return 0.0
    return float(max(client_f1_list) - min(client_f1_list))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 단위 테스트 (assert 구문)
# ─────────────────────────────────────────────────────────────────────────────
def _run_unit_tests() -> None:
    """Dummy 예측값으로 두 함수의 정확성을 검증한다."""

    print("=" * 55)
    print("metrics.py 단위 테스트 시작")
    print("=" * 55)

    # ── Test A: calculate_local_macro_f1 ──────────────────────────
    # 클래스 0, 1, 2 만 존재; 클래스 3은 부재
    y_true_a = np.array([0, 0, 1, 1, 2, 2])
    y_pred_a = np.array([0, 1, 1, 1, 2, 0])  # class0: tp=1,fp=1,fn=1 / class1: tp=2,fp=1,fn=0 / class2: tp=1,fp=0,fn=1

    # 수동 계산 (y_true=[0,0,1,1,2,2], y_pred=[0,1,1,1,2,0]):
    # Class 0: TP=1 (idx0), FP=1 (idx5: pred=0,true=2), FN=1 (idx1: pred=1,true=0)
    #          → F1 = 2*1/(2+1+1) = 0.5
    # Class 1: TP=2 (idx2,3), FP=1 (idx1: pred=1,true=0), FN=0
    #          → F1 = 4/(4+1+0) = 0.8
    # Class 2: TP=1 (idx4), FP=0, FN=1 (idx5: pred=0,true=2)
    #          → F1 = 2/(2+0+1) = 2/3
    # Macro-F1 = (0.5 + 0.8 + 2/3) / 3
    expected_f1_a = (0.5 + 0.8 + 2/3) / 3
    result_f1_a   = calculate_local_macro_f1(y_true_a, y_pred_a,
                                              present_classes=[0, 1, 2])
    assert abs(result_f1_a - expected_f1_a) < 1e-9, (
        f"[FAIL] Macro-F1: expected {expected_f1_a:.6f}, got {result_f1_a:.6f}"
    )
    print(f"[PASS] Test A-1: Macro-F1 = {result_f1_a:.6f} (expected {expected_f1_a:.6f})")

    # 클래스 3(부재)을 present_classes에 포함하지 않으면 결과 동일해야 함
    result_f1_b = calculate_local_macro_f1(y_true_a, y_pred_a,
                                            present_classes=[0, 1, 2])
    assert abs(result_f1_a - result_f1_b) < 1e-9, "[FAIL] present_classes 필터링 불일치"
    print("[PASS] Test A-2: present_classes 필터링 일관성 확인")

    # present_classes 가 비어있으면 0.0
    result_empty = calculate_local_macro_f1(y_true_a, y_pred_a,
                                             present_classes=[])
    assert result_empty == 0.0, "[FAIL] 빈 present_classes → 0.0 기대"
    print("[PASS] Test A-3: 빈 present_classes → 0.0")

    # 완벽한 예측 → F1 = 1.0
    y_perfect = np.array([0, 1, 2, 3])
    result_perfect = calculate_local_macro_f1(y_perfect, y_perfect,
                                               present_classes=[0, 1, 2, 3])
    assert abs(result_perfect - 1.0) < 1e-9, "[FAIL] 완벽 예측 → F1=1.0 기대"
    print("[PASS] Test A-4: 완벽 예측 → Macro-F1 = 1.0")

    # ── Test B: calculate_ece ──────────────────────────────────────
    # 케이스 1: 완벽 보정 – confidence = accuracy in each bin → ECE ≈ 0
    # 10개 샘플, 각 샘플이 정확히 [0.05, 0.15, ..., 0.95] confidence 를 갖고
    # confidence 구간 내에서 해당 비율만큼 정답이면 ECE = 0
    n_ece = 1000
    rng   = np.random.default_rng(0)
    confs = rng.uniform(0, 1, n_ece)
    # 각 샘플을 confidence 에 비례한 확률로 정답 처리 → 완벽 보정 근사
    correct_mask = rng.uniform(0, 1, n_ece) < confs
    y_true_ece   = np.where(correct_mask, 0, 1)
    y_prob_ece   = np.zeros((n_ece, 2))
    y_prob_ece[:, 0] = confs
    y_prob_ece[:, 1] = 1 - confs
    ece_perfect  = calculate_ece(y_true_ece, y_prob_ece)
    # 랜덤 근사이므로 완벽하진 않지만 0.05 이내여야 함
    assert ece_perfect < 0.05, (
        f"[FAIL] 완벽 보정 근사 ECE={ece_perfect:.4f} > 0.05"
    )
    print(f"[PASS] Test B-1: 완벽 보정 근사 ECE = {ece_perfect:.4f} < 0.05")

    # 케이스 2: 완전 오예측 (confidence=1.0, 항상 틀림) → ECE = 1.0
    y_true_c2 = np.array([0, 0, 0])
    y_prob_c2 = np.array([[0.0, 1.0],   # 예측 클래스 1, 정답 0 → 틀림
                           [0.0, 1.0],
                           [0.0, 1.0]])
    ece_c2 = calculate_ece(y_true_c2, y_prob_c2)
    assert abs(ece_c2 - 1.0) < 1e-9, (
        f"[FAIL] 완전 오예측 ECE={ece_c2:.6f} ≠ 1.0"
    )
    print(f"[PASS] Test B-2: 완전 오예측 ECE = {ece_c2:.6f} (expected 1.0)")

    # 케이스 3: 완벽 예측 (confidence=1.0, 항상 맞음) → ECE = 0.0
    y_true_c3 = np.array([0, 1, 0])
    y_prob_c3 = np.array([[1.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 0.0]])
    ece_c3 = calculate_ece(y_true_c3, y_prob_c3)
    assert abs(ece_c3 - 0.0) < 1e-9, (
        f"[FAIL] 완벽 예측 ECE={ece_c3:.6f} ≠ 0.0"
    )
    print(f"[PASS] Test B-3: 완벽 예측 ECE = {ece_c3:.6f} (expected 0.0)")

    # 케이스 4: 수동 구간 검증
    # y_true=[0,1], y_prob=[[0.85,0.15],[0.90,0.10]] → 둘 다 클래스 0 예측
    #   sample 0: conf=0.85 → 구간 [0.8,0.9), correct=1 (true=0,pred=0)
    #   sample 1: conf=0.90 → 구간 [0.9,1.0], correct=0 (true=1,pred=0)
    #   구간 [0.8,0.9): acc=1.0, conf=0.85 → |1.0-0.85|=0.15, weight=1/2
    #   구간 [0.9,1.0]: acc=0.0, conf=0.90 → |0.0-0.90|=0.90, weight=1/2
    #   ECE = 0.5*0.15 + 0.5*0.90 = 0.075 + 0.45 = 0.525
    y_true_c4 = np.array([0, 1])
    y_prob_c4 = np.array([[0.85, 0.15],
                           [0.90, 0.10]])  # 둘 다 클래스 0 예측
    ece_c4    = calculate_ece(y_true_c4, y_prob_c4, num_bins=10)
    expected_ece_c4 = 0.5 * abs(1.0 - 0.85) + 0.5 * abs(0.0 - 0.90)   # = 0.525
    assert abs(ece_c4 - expected_ece_c4) < 1e-9, (
        f"[FAIL] 수동 구간 ECE={ece_c4:.6f} ≠ {expected_ece_c4:.6f}"
    )
    print(f"[PASS] Test B-4: 수동 구간 검증 ECE = {ece_c4:.6f} "
          f"(expected {expected_ece_c4:.6f})")

    # ── Test C: calculate_f1_gap ───────────────────────────────────
    gap = calculate_f1_gap([0.9, 0.6, 0.75])
    assert abs(gap - 0.3) < 1e-9, f"[FAIL] Gap={gap:.6f} ≠ 0.3"
    print(f"[PASS] Test C-1: F1 Gap = {gap:.6f} (expected 0.3)")

    gap_eq = calculate_f1_gap([0.8, 0.8, 0.8])
    assert abs(gap_eq - 0.0) < 1e-9, f"[FAIL] Gap(equal)={gap_eq:.6f} ≠ 0.0"
    print(f"[PASS] Test C-2: F1 Gap (equal) = {gap_eq:.6f} (expected 0.0)")

    print("\n[OK] 모든 단위 테스트 통과.")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _run_unit_tests()
