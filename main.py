import argparse
import copy
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn

from data_utils import compute_class_weights, get_dataloaders
from federated_core import (
    build_efficientnet_b0,
    compute_model_size_mb,
    compute_round_comm_mb,
    compute_weight_divergence,
    evaluate_model,
    server_aggregate,
    train_one_client,
)


CSV_FIELDS = [
    "round",
    "method",
    "mode",
    "mu",
    "seed",
    "train_loss",
    "test_loss",
    "balanced_acc",
    "macro_f1",
    "ece",
    "max_macro_f1",
    "min_macro_f1",
    "macro_f1_gap",
    "weight_divergence",
    "cumulative_comm_mb",
]

# resume 시 기존 체크포인트와 반드시 일치해야 하는 실험 설정 키
RESUME_CONFIG_KEYS = ["method", "mode", "mu", "seed", "lr", "local_epochs", "batch_size"]


def set_seed(seed: int):
    """실험 재현성을 위해 Python, NumPy, PyTorch(CPU/GPU) 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fed-ISIC2019 Federated Learning (Centralized / FedAvg / FedProx)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        choices=["centralized", "fedavg", "fedprox"],
        help="학습 방식 선택: centralized, fedavg, fedprox",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="top3",
        choices=["top3", "merge6"],
        help="데이터셋 분할 모드: top3 (센터 0,1,2) 또는 merge6 (6개 센터 3분할 병합)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="총 통신 라운드 수 (centralized의 경우 총 에폭 단위로 동작)",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=2,
        help="각 라운드당 클라이언트 로컬 에폭 수 (연합학습 권장: 2, centralized 권장: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="미니배치 크기",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="학습률 (Learning Rate)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx 근접 항 계수 (method='fedprox'일 때 적용, fedavg/centralized는 0.0으로 강제)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="로그 CSV 및 최고 성능 모델 가중치(.pt) 저장 디렉토리",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="전처리된 Fed-ISIC2019 데이터 경로 (ISIC_2019_Training_Input_preprocessed 폴더의 상위 경로). "
             "None이면 FLamby의 dataset_location.yaml 설정을 사용",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers 수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성 및 다중 시드 실험을 위한 난수 시드 (기본값: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="save_dir에 동일 실험의 last 체크포인트가 있으면 마지막 완료 라운드 다음부터 이어서 학습",
    )
    return parser.parse_args()


def _atomic_torch_save(obj, path: str):
    """저장 도중 세션이 끊겨도 기존 파일이 손상되지 않도록 임시 파일에 저장 후 교체합니다."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _capture_rng_states(client_loaders, centralized_loader) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "client_loader_generators": [
            l.generator.get_state() if l.generator is not None else None for l in client_loaders
        ],
        "central_loader_generator": (
            centralized_loader.generator.get_state()
            if centralized_loader.generator is not None
            else None
        ),
    }


def _restore_rng_states(states: dict, client_loaders, centralized_loader):
    try:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"].cpu())
        if torch.cuda.is_available() and states.get("cuda") is not None:
            cuda_states = [s.cpu() for s in states["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        for loader, g_state in zip(client_loaders, states.get("client_loader_generators", [])):
            if loader.generator is not None and g_state is not None:
                loader.generator.set_state(g_state.cpu())
        c_state = states.get("central_loader_generator")
        if centralized_loader.generator is not None and c_state is not None:
            centralized_loader.generator.set_state(c_state.cpu())
    except Exception as e:
        print(f"[Resume Warning] 난수 상태 일부 복원 생략 (GPU/환경 변경 가능성): {e}")


def _truncate_csv(csv_path: str, last_round: int):
    """resume 시 체크포인트 이후에 기록된(중복될) CSV 행을 제거합니다."""
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if int(row["round"]) <= last_round]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    effective_mu = args.mu if args.method == "fedprox" else 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device} | Seed: {args.seed}")
    print(
        f"[Info] Method: {args.method} | Mode: {args.mode} | Rounds: {args.rounds} | "
        f"Local Epochs: {args.local_epochs} | Batch Size: {args.batch_size} | "
        f"LR: {args.lr} | Mu: {effective_mu}"
    )

    # 1. DataLoader 준비
    client_loaders, centralized_loader, test_loader = get_dataloaders(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_path=args.data_path,
        seed=args.seed,
    )

    num_clients = len(client_loaders)
    client_sample_counts = [len(loader.dataset) for loader in client_loaders]
    print(f"[Info] Number of clients: {num_clients} | Client samples: {client_sample_counts}")
    print(f"[Info] Centralized train samples: {len(centralized_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # 2. 클래스 불균형 해소를 위한 역빈도 가중치 계산
    if args.method == "centralized":
        class_weights = compute_class_weights(centralized_loader, num_classes=8, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        client_criteria = None
    else:
        # 각 클라이언트의 로컬 클래스 분포에 맞춘 역빈도 가중치 (전체 통합 가중치로 평가)
        client_criteria = [
            nn.CrossEntropyLoss(
                weight=compute_class_weights(loader, num_classes=8, device=device)
            )
            for loader in client_loaders
        ]
        eval_weights = compute_class_weights(centralized_loader, num_classes=8, device=device)
        criterion = nn.CrossEntropyLoss(weight=eval_weights)

    # 3. 글로벌 모델 초기화 및 1라운드당 통신량(MB) 계산
    global_model = build_efficientnet_b0(num_classes=8, pretrained=True).to(device)
    model_size_mb = compute_model_size_mb(global_model)
    round_comm_mb = (
        0.0
        if args.method == "centralized"
        else compute_round_comm_mb(global_model, num_clients)
    )
    print(
        f"[Info] Model Size: {model_size_mb:.2f} MB | "
        f"Comm Volume per Round (Down+Up): {round_comm_mb:.2f} MB"
    )

    # Centralized는 에폭 간 Adam 모멘텀을 유지해야 하므로 옵티마이저를 루프 밖에서 1회만 생성
    # (연합학습은 라운드마다 클라이언트 로컬 옵티마이저를 새로 생성하는 표준 방식을 따름)
    central_optimizer = (
        torch.optim.Adam(global_model.parameters(), lr=args.lr)
        if args.method == "centralized"
        else None
    )

    # 4. 결과 기록용 CSV 및 모델 저장 경로 설정 (시드 및 FedProx mu 값 구분)
    if args.method == "fedprox":
        exp_tag = f"{args.method}_mu{effective_mu:g}_{args.mode}"
    else:
        exp_tag = f"{args.method}_{args.mode}"

    csv_path = os.path.join(args.save_dir, f"{exp_tag}_seed{args.seed}_metrics.csv")
    best_model_path = os.path.join(args.save_dir, f"best_{exp_tag}_seed{args.seed}.pt")
    last_ckpt_path = os.path.join(args.save_dir, f"last_{exp_tag}_seed{args.seed}.pt")

    run_config = {
        "method": args.method,
        "mode": args.mode,
        "mu": effective_mu,
        "seed": args.seed,
        "lr": args.lr,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
    }

    start_round = 1
    best_balanced_acc = -1.0

    # 5. Resume: 동일 실험의 last 체크포인트가 있으면 이어서 학습
    if args.resume and os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(last_ckpt_path, map_location="cpu")
        saved_config = ckpt.get("config", {})
        mismatched = [
            k for k in RESUME_CONFIG_KEYS if saved_config.get(k) != run_config[k]
        ]
        if mismatched:
            raise ValueError(
                f"[Resume] 체크포인트 설정과 현재 인자가 다릅니다: "
                + ", ".join(f"{k} (ckpt={saved_config.get(k)}, now={run_config[k]})" for k in mismatched)
            )

        global_model.load_state_dict(ckpt["model_state_dict"])
        global_model.to(device)
        if central_optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            central_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_balanced_acc = ckpt["best_balanced_acc"]
        start_round = ckpt["round"] + 1
        _restore_rng_states(ckpt["rng_states"], client_loaders, centralized_loader)
        _truncate_csv(csv_path, ckpt["round"])
        print(
            f"[Resume] {last_ckpt_path} 로드 완료 → Round {start_round}부터 재개 "
            f"(현재 Best Balanced Acc: {best_balanced_acc:.4f})"
        )
        if start_round > args.rounds:
            print(f"[Resume] 이미 {ckpt['round']} 라운드까지 완료된 실험입니다. 종료합니다.")
            return
    else:
        if args.resume:
            print("[Resume] 기존 체크포인트가 없어 처음부터 학습합니다.")
        elif os.path.exists(last_ckpt_path):
            print(f"[Warning] 기존 체크포인트({last_ckpt_path})를 무시하고 처음부터 학습합니다. 이어서 학습하려면 --resume을 사용하세요.")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    # 6. 라운드별 학습 및 평가 루프
    for r in range(start_round, args.rounds + 1):
        if args.method == "centralized":
            global_model, train_loss = train_one_client(
                model=global_model,
                dataloader=centralized_loader,
                criterion=criterion,
                lr=args.lr,
                local_epochs=args.local_epochs,
                device=device,
                mu=0.0,
                global_model=None,
                optimizer=central_optimizer,
            )
            # Centralized는 단일 글로벌 모델만 존재하므로 클라이언트 간 가중치 발산(Client Drift)은 0.0
            avg_divergence = 0.0
        else:
            local_models = []
            local_losses = []
            divergences = []

            for client_idx, loader in enumerate(client_loaders):
                local_model = copy.deepcopy(global_model)
                trained_local_model, c_loss = train_one_client(
                    model=local_model,
                    dataloader=loader,
                    criterion=client_criteria[client_idx],
                    lr=args.lr,
                    local_epochs=args.local_epochs,
                    device=device,
                    mu=effective_mu,
                    global_model=global_model,
                )

                div = compute_weight_divergence(global_model, trained_local_model)
                local_models.append(trained_local_model)
                local_losses.append(c_loss)
                divergences.append(div)

            # 샘플 수 비례 가중 평균으로 글로벌 모델 업데이트 및 평균 손실(CE 기준) 계산
            total_samples = sum(client_sample_counts)
            train_loss = sum(
                l * n for l, n in zip(local_losses, client_sample_counts)
            ) / max(total_samples, 1)
            avg_divergence = sum(divergences) / max(len(divergences), 1)

            global_model = server_aggregate(
                global_model=global_model,
                client_models=local_models,
                client_sample_counts=client_sample_counts,
            )

        # 글로벌 테스트셋 및 클라이언트별 테스트셋 평가
        eval_metrics = evaluate_model(
            model=global_model,
            dataloader=test_loader,
            device=device,
            criterion=criterion,
        )
        test_loss = eval_metrics["loss"]
        balanced_acc = eval_metrics["balanced_acc"]
        macro_f1 = eval_metrics["macro_f1"]
        ece = eval_metrics["ece"]
        max_macro_f1 = eval_metrics["max_macro_f1"]
        min_macro_f1 = eval_metrics["min_macro_f1"]
        macro_f1_gap = eval_metrics["macro_f1_gap"]
        cumulative_comm_mb = r * round_comm_mb

        print(
            f"[Round {r:03d}/{args.rounds:03d}] "
            f"Train CE: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Bal Acc: {balanced_acc:.4f} | Macro F1: {macro_f1:.4f} "
            f"(Gap: {macro_f1_gap:.4f} [{min_macro_f1:.4f}~{max_macro_f1:.4f}]) | "
            f"ECE: {ece:.4f} | Weight Div: {avg_divergence:.4f} | "
            f"Comm: {cumulative_comm_mb:.1f} MB"
        )

        # 매 라운드 결과를 실시간으로 CSV 파일에 기록 (flush 보장)
        row_data = {
            "round": r,
            "method": args.method,
            "mode": args.mode,
            "mu": effective_mu,
            "seed": args.seed,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6),
            "balanced_acc": round(balanced_acc, 6),
            "macro_f1": round(macro_f1, 6),
            "ece": round(ece, 6),
            "max_macro_f1": round(max_macro_f1, 6),
            "min_macro_f1": round(min_macro_f1, 6),
            "macro_f1_gap": round(macro_f1_gap, 6),
            "weight_divergence": round(avg_divergence, 6),
            "cumulative_comm_mb": round(cumulative_comm_mb, 2),
        }
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow(row_data)
            f.flush()

        # 최고 balanced_acc 달성 시 .pt 가중치 저장
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            _atomic_torch_save(
                {
                    "round": r,
                    "config": run_config,
                    "balanced_acc": balanced_acc,
                    "macro_f1": macro_f1,
                    "ece": ece,
                    "max_macro_f1": max_macro_f1,
                    "min_macro_f1": min_macro_f1,
                    "macro_f1_gap": macro_f1_gap,
                    "cumulative_comm_mb": cumulative_comm_mb,
                    "model_state_dict": global_model.state_dict(),
                },
                best_model_path,
            )
            print(
                f"  -> [*] Best Balanced Acc 갱신 ({best_balanced_acc:.4f})! 가중치 저장 완료: {best_model_path}"
            )

        # 매 라운드 종료 시 resume용 last 체크포인트 저장 (모델 + 옵티마이저 + 난수 상태)
        _atomic_torch_save(
            {
                "round": r,
                "config": run_config,
                "best_balanced_acc": best_balanced_acc,
                "model_state_dict": global_model.state_dict(),
                "optimizer_state_dict": (
                    central_optimizer.state_dict() if central_optimizer is not None else None
                ),
                "rng_states": _capture_rng_states(client_loaders, centralized_loader),
            },
            last_ckpt_path,
        )

    print(
        f"[Done] 학습 완료. 최고 Balanced Acc: {best_balanced_acc:.4f} | 결과 파일: {csv_path}"
    )


if __name__ == "__main__":
    main()
