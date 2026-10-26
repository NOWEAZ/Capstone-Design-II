import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from flamby.datasets.fed_isic2019 import FedIsic2019


def _seed_worker(worker_id: int):
    """DataLoader 멀티프로세싱 워커의 난수 시드를 고정합니다."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _extract_labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """Dataset(또는 ConcatDataset)에서 이미지 디코딩 없이 빠르게 레이블 배열을 추출합니다."""
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_extract_labels_from_dataset(d) for d in dataset.datasets])

    # FLamby FedIsic2019 내부 속성 확인 (버전별 호환)
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            vals = getattr(dataset, attr)
            return np.asarray(vals, dtype=np.int64).reshape(-1)

    if hasattr(dataset, "dic") and isinstance(dataset.dic, dict) and "target" in dataset.dic:
        return np.asarray(dataset.dic["target"], dtype=np.int64).reshape(-1)

    # Fallback: 데이터셋 직접 순회
    labels = []
    for _, y in dataset:
        if isinstance(y, torch.Tensor):
            labels.append(int(y.item()))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def compute_class_weights(
    data_source,
    num_classes: int = 8,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    역빈도(Inverse-Frequency) 기반 클래스 가중치를 계산합니다.
    w_c = N / (K * n_c)

    Args:
        data_source: Dataset, DataLoader, 또는 이들의 리스트/반복자, 혹은 레이블 배열
        num_classes: 전체 클래스 개수 (Fed-ISIC2019 기본값: 8)
        device: 반환할 텐서의 디바이스

    Returns:
        torch.Tensor: shape (num_classes,), dtype=torch.float32
    """
    if isinstance(data_source, DataLoader):
        labels = _extract_labels_from_dataset(data_source.dataset)
    elif isinstance(data_source, Dataset):
        labels = _extract_labels_from_dataset(data_source)
    elif isinstance(data_source, (list, tuple)):
        extracted = []
        for item in data_source:
            if isinstance(item, DataLoader):
                extracted.append(_extract_labels_from_dataset(item.dataset))
            elif isinstance(item, Dataset):
                extracted.append(_extract_labels_from_dataset(item))
            else:
                extracted.append(np.asarray(item, dtype=np.int64).reshape(-1))
        labels = np.concatenate(extracted)
    else:
        labels = np.asarray(data_source, dtype=np.int64).reshape(-1)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total_samples = float(len(labels))

    # 특정 클래스 샘플 수가 0인 경우 division by zero 방지
    safe_counts = np.where(counts > 0, counts, 1.0)
    weights = total_samples / (num_classes * safe_counts)
    weights = np.where(counts > 0, weights, 0.0)

    # 가중치 평균이 1이 되도록 정규화
    if weights.sum() > 0:
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_dataloaders(
    mode: str = "top3",
    batch_size: int = 32,
    num_workers: int = 4,
    data_path: str | None = None,
    debug: bool = False,
    seed: int | None = None,
):
    """
    FLamby의 FedIsic2019 데이터셋을 사용해 클라이언트별 DataLoader,
    Centralized 통합 DataLoader, Global Test DataLoader를 생성합니다.

    Args:
        mode: 'top3' (센터 0, 1, 2 사용) 또는 'merge6' (6개 센터를 3개 클라이언트로 병합)
        batch_size: 미니배치 크기
        num_workers: DataLoader 워커 프로세스 수
        data_path: FLamby 데이터 경로 (None이면 FLamby 기본 설정 사용)
        debug: FLamby debug 모드 여부
        seed: 재현성을 위한 난수 시드 (None이면 고정하지 않음)

    Returns:
        tuple: (client_loaders, centralized_loader, test_loader)
            - client_loaders ( list[DataLoader] ): 3개 클라이언트의 학습용 DataLoader 리스트
            - centralized_loader ( DataLoader ): 전체 학습 데이터를 하나로 묶은 통합 DataLoader
            - test_loader ( DataLoader ): 글로벌 평가용 통합 Test DataLoader
    """
    mode = mode.lower().strip()
    if mode not in ("top3", "merge6"):
        raise ValueError(f"지원하지 않는 mode입니다: {mode}. 'top3' 또는 'merge6'을 사용하세요.")

    flamby_kwargs = {"debug": debug}
    if data_path is not None:
        flamby_kwargs["data_path"] = data_path

    if mode == "top3":
        # 상위 3개 주요 센터 (센터 0, 1, 2)
        client_datasets = [
            FedIsic2019(center=i, train=True, pooled=False, **flamby_kwargs)
            for i in range(3)
        ]
        client_test_datasets = [
            FedIsic2019(center=i, train=False, pooled=False, **flamby_kwargs)
            for i in range(3)
        ]
    else:
        # mode == 'merge6': 전체 6개 센터(0~5)를 3개 클라이언트로 병합
        # 대형 센터(0, 1, 2) 각각에 소형 센터(3, 4, 5)를 페어링하여 3개 클라이언트로 구성
        center_groups = [(0, 3), (1, 4), (2, 5)]
        client_datasets = []
        client_test_datasets = []
        for group in center_groups:
            merged_train = ConcatDataset(
                [
                    FedIsic2019(center=c, train=True, pooled=False, **flamby_kwargs)
                    for c in group
                ]
            )
            merged_test = ConcatDataset(
                [
                    FedIsic2019(center=c, train=False, pooled=False, **flamby_kwargs)
                    for c in group
                ]
            )
            client_datasets.append(merged_train)
            client_test_datasets.append(merged_test)

    pin_memory = torch.cuda.is_available()
    worker_init_fn = _seed_worker if seed is not None else None

    # 1. 3개 클라이언트 학습 DataLoader
    client_loaders = []
    for idx, ds in enumerate(client_datasets):
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed + idx)
        client_loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                worker_init_fn=worker_init_fn,
                generator=gen,
            )
        )

    # 2. Centralized 통합 학습 DataLoader
    centralized_dataset = ConcatDataset(client_datasets)
    central_gen = None
    if seed is not None:
        central_gen = torch.Generator()
        central_gen.manual_seed(seed + 100)
    centralized_loader = DataLoader(
        centralized_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=central_gen,
    )

    # 3. Global Test DataLoader (전체 센터 통합 테스트셋) 및 클라이언트별 Test DataLoader
    test_dataset = FedIsic2019(train=False, pooled=True, **flamby_kwargs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # 클라이언트 간 최고-최저 Macro-F1(공정성 지표) 계산을 위해 클라이언트별 테스트 로더를 속성으로 연결
    test_loader.client_test_loaders = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        for ds in client_test_datasets
    ]

    return client_loaders, centralized_loader, test_loader

