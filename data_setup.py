"""
data_setup.py  –  1주 차: BloodMNIST 데이터 분할 및 시각화
=============================================================
README 제약:
  - 로컬 Test 셋 생성 금지 (Validation 분할만 제공)
  - IID: 클래스별 층화 균등 배분
  - non-IID: Dirichlet(alpha) 기반 largest-remainder 정수 배분
  - 최소 표본 1,000장 미만 시 최대 100회 재추첨
  - 전처리: Normalize([0.5]*3, [0.5]*3), 증강/오버샘플링 금지
"""

import random
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import Subset
from medmnist import BloodMNIST
import torchvision.transforms as transforms

# ── 전역 상수 ────────────────────────────────────────────────────────────────
NUM_CLIENTS   = 3
NUM_CLASSES   = 8
MIN_SAMPLES   = 1_000
MAX_RETRIES   = 100
SEEDS         = [40, 41, 42, 43, 44]
ALPHA_LIST    = [0.5, 0.1]
OUTPUT_DIR    = "outputs/data_setup"
CLASS_NAMES   = [
    "basophil", "eosinophil", "erythroblast",
    "ig",       "lymphocyte",  "monocyte",
    "neutrophil", "platelet",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 공통 전처리 Transform (README §3.4: [0,1] 후 mean=0.5, std=0.5 고정)
# ─────────────────────────────────────────────────────────────────────────────
def get_transform() -> transforms.Compose:
    """증강·오버샘플링 없이 정규화만 수행."""
    return transforms.Compose([
        transforms.ToTensor(),                          # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터 로드 (Train / Validation 분리)
# ─────────────────────────────────────────────────────────────────────────────
def load_bloodmnist(root: str = "./data"):
    """
    BloodMNIST를 다운로드하고 Train / Validation Dataset 객체를 반환한다.
    로컬 Test 셋은 생성하지 않는다 (README §3.3).
    """
    tf = get_transform()
    train_ds = BloodMNIST(split="train",      transform=tf,
                          download=True, root=root)
    val_ds   = BloodMNIST(split="val",        transform=tf,
                          download=True, root=root)
    # NOTE: "test" 분할은 서버 측 통제 하에만 존재 — 여기서는 절대 로드 안 함.
    return train_ds, val_ds


def get_labels(dataset) -> np.ndarray:
    """Dataset에서 정수 레이블 배열을 추출."""
    return np.array(dataset.labels).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# 3. IID 분할 – 클래스별 층화 균등 배분
# ─────────────────────────────────────────────────────────────────────────────
def partition_iid(labels: np.ndarray,
                  n_clients: int,
                  rng: np.random.Generator) -> list[list[int]]:
    """
    각 클래스 내에서 인덱스를 셔플한 뒤 n_clients에 균등 배분한다.
    surplus(나머지)는 앞쪽 클라이언트부터 1장씩 배분.
    """
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0].tolist()
        rng.shuffle(idx)
        base, rem = divmod(len(idx), n_clients)
        ptr = 0
        for cid in range(n_clients):
            count = base + (1 if cid < rem else 0)
            client_indices[cid].extend(idx[ptr: ptr + count])
            ptr += count

    # 클라이언트 내부 인덱스도 셔플
    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# 4. non-IID 분할 – Dirichlet + largest-remainder 정수 배분
# ─────────────────────────────────────────────────────────────────────────────
def _largest_remainder(proportions: np.ndarray, total: int) -> list[int]:
    """
    비율 배열을 정수 카운트로 변환 (합 = total).
    surplus 표본은 소수점 크기 내림차순(largest-remainder) 기준 배분.
    """
    floors    = np.floor(proportions * total).astype(int)
    remainder = total - floors.sum()
    residuals = (proportions * total) - floors
    # largest-remainder: 소수점이 큰 클라이언트 순서대로 +1
    order = np.argsort(-residuals)
    for i in range(int(remainder)):
        floors[order[i]] += 1
    return floors.tolist()


def partition_noniid(labels: np.ndarray,
                     n_clients: int,
                     alpha: float,
                     rng: np.random.Generator) -> list[list[int]]:
    """
    Dirichlet(alpha) 분포로 각 클래스 내 클라이언트 비율을 샘플링한 뒤
    largest-remainder 정수 배분을 적용한다.
    남는 표본(잔여 LR 동률)은 시드 순서(=클라이언트 인덱스 오름차순)로 배분.
    """
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0].tolist()
        rng.shuffle(idx)
        n_c = len(idx)

        # Dirichlet 비율 샘플링
        props = rng.dirichlet(alpha=np.full(n_clients, alpha))

        # largest-remainder 정수 배분
        counts = _largest_remainder(props, n_c)

        ptr = 0
        for cid in range(n_clients):
            client_indices[cid].extend(idx[ptr: ptr + counts[cid]])
            ptr += counts[cid]

    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# 5. 최소 표본 검증 루프 (README §3.2)
# ─────────────────────────────────────────────────────────────────────────────
def partition_with_retry(
    labels: np.ndarray,
    mode: str,          # "iid" | "noniid"
    seed: int,
    alpha: float = 0.5,
    n_clients: int = NUM_CLIENTS,
    min_samples: int = MIN_SAMPLES,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[list[int]], int]:
    """
    분할을 시도하고, 클라이언트 중 min_samples 미만이 있으면 재시도.
    반환: (client_indices, 실제_시도_횟수)
    최대 max_retries 이후에도 실패하면 마지막 결과를 반환하고 경고 출력.
    """
    attempt  = 0
    base_rng = np.random.default_rng(seed)

    while attempt < max_retries:
        # 시도마다 독립적인 RNG 상태 파생 (시도 인덱스로 spawn)
        child_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

        if mode == "iid":
            indices = partition_iid(labels, n_clients, child_rng)
        else:
            indices = partition_noniid(labels, n_clients, alpha, child_rng)

        sizes = [len(idx) for idx in indices]
        attempt += 1

        if all(s >= min_samples for s in sizes):
            return indices, attempt

    # 최대 재시도 초과
    print(f"  [WARNING] seed={seed}, mode={mode}, alpha={alpha}: "
          f"최대 {max_retries}회 시도 후에도 min_samples={min_samples} 미충족. "
          f"마지막 결과 사용 (sizes={sizes}).")
    return indices, attempt  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# 6. 시각화 및 통계
# ─────────────────────────────────────────────────────────────────────────────
def _class_dist(indices: list[int], labels: np.ndarray) -> list[int]:
    """인덱스 리스트 → 클래스별 카운트 (길이 NUM_CLASSES)."""
    cnt = Counter(labels[indices].tolist())
    return [cnt.get(c, 0) for c in range(NUM_CLASSES)]


def visualize_distribution(
    client_indices: list[list[int]],
    labels: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """클라이언트별 클래스 분포 누적 바 차트 저장."""
    n = len(client_indices)
    dists = np.array([_class_dist(idx, labels) for idx in client_indices],
                     dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = 0.5
    x = np.arange(n)
    bottom = np.zeros(n)
    cmap = plt.get_cmap("tab10")

    for c in range(NUM_CLASSES):
        bars = ax.bar(x, dists[:, c], bar_width, bottom=bottom,
                      label=CLASS_NAMES[c], color=cmap(c))
        # 각 구간 중앙에 카운트 표시 (너무 작으면 생략)
        for bar, bot in zip(bars, bottom):
            h = bar.get_height()
            if h > 50:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + h / 2, f"{int(h)}",
                        ha="center", va="center", fontsize=6, color="white")
        bottom += dists[:, c]

    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {i}" for i in range(n)])
    ax.set_ylabel("# Samples")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 시각화 저장: {save_path}")


def compute_pass_rate(
    labels: np.ndarray,
    seed: int,
    mode: str,
    alpha: float = 0.5,
    trials: int = 200,
) -> float:
    """
    동일 조건으로 trials 회 독립 분할 시도 시 min_samples 통과율을 반환.
    (최소 표본 통과율 계산용 헬퍼)
    """
    passed = 0
    base_rng = np.random.default_rng(seed + 9999)  # 메인 루프와 독립
    for _ in range(trials):
        child_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        if mode == "iid":
            indices = partition_iid(labels, NUM_CLIENTS, child_rng)
        else:
            indices = partition_noniid(labels, NUM_CLIENTS, alpha, child_rng)
        if all(len(idx) >= MIN_SAMPLES for idx in indices):
            passed += 1
    return passed / trials


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 실행 – 모든 시드 × 모드 조합 수행
# ─────────────────────────────────────────────────────────────────────────────
def run_all_partitions(train_ds, val_ds) -> dict:
    """
    SEEDS × {iid, noniid(0.5), noniid(0.1)} 를 모두 수행하고
    결과 딕셔너리를 반환한다.
    """
    train_labels = get_labels(train_ds)
    val_labels   = get_labels(val_ds)

    results: dict = {}

    configs = (
        [("iid", None)] +
        [("noniid", a) for a in ALPHA_LIST]
    )

    for seed in SEEDS:
        results[seed] = {}
        for (mode, alpha) in configs:
            key = mode if alpha is None else f"{mode}_alpha{alpha}"
            print(f"\n[Seed={seed} | {key}]")

            indices, attempts = partition_with_retry(
                train_labels, mode=mode, seed=seed,
                alpha=alpha if alpha is not None else 0.5,
            )
            sizes = [len(idx) for idx in indices]
            print(f"  시도 횟수: {attempts}  |  Client sizes: {sizes}")

            # 최소 표본 통과율 (빠른 추정: 100 trials)
            pass_rate = compute_pass_rate(
                train_labels, seed=seed, mode=mode,
                alpha=alpha if alpha is not None else 0.5,
                trials=100,
            )
            print(f"  통과율 추정({100} trials): {pass_rate*100:.1f}%")

            results[seed][key] = {
                "indices":    indices,
                "sizes":      sizes,
                "attempts":   attempts,
                "pass_rate":  pass_rate,
            }

            # 시각화
            title_alpha = "" if alpha is None else f", α={alpha}"
            title = (f"BloodMNIST Client Distribution  "
                     f"[{key.upper()}{title_alpha}]  seed={seed}")
            save_path = os.path.join(
                OUTPUT_DIR, f"dist_{key}_seed{seed}.png")
            visualize_distribution(indices, train_labels, title, save_path)

    # 통계 요약 출력
    print("\n" + "="*60)
    print("분할 통계 요약")
    print("="*60)
    for seed in SEEDS:
        for key, info in results[seed].items():
            sizes     = info["sizes"]
            pass_rate = info["pass_rate"]
            print(f"  seed={seed} | {key:<25} "
                  f"sizes={[f'{s:,}' for s in sizes]}  "
                  f"pass_rate={pass_rate*100:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. Subset 헬퍼 – 후속 파이프라인에서 DataLoader 생성용
# ─────────────────────────────────────────────────────────────────────────────
def make_client_subsets(
    train_ds,
    client_indices: list[list[int]],
) -> list[Subset]:
    """인덱스 리스트 → torch Subset 리스트 반환."""
    return [Subset(train_ds, idx) for idx in client_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("BloodMNIST 데이터 로드 중...")
    train_ds, val_ds = load_bloodmnist()
    print(f"  Train: {len(train_ds):,}장  |  Val: {len(val_ds):,}장")
    print(f"  출력 디렉터리: {os.path.abspath(OUTPUT_DIR)}\n")

    results = run_all_partitions(train_ds, val_ds)
    print("\n✓ data_setup.py 완료.")

