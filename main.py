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
    return parser.parse_args()


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

    # 4. 결과 기록용 CSV 및 모델 저장 경로 설정 (시드 및 FedProx mu 값 구분)
    if args.method == "fedprox":
        exp_tag = f"{args.method}_mu{effective_mu:g}_{args.mode}"
    else:
        exp_tag = f"{args.method}_{args.mode}"

    csv_filename = f"{exp_tag}_seed{args.seed}_metrics.csv"
    csv_path = os.path.join(args.save_dir, csv_filename)
    best_model_path = os.path.join(args.save_dir, f"best_{exp_tag}_seed{args.seed}.pt")

    csv_fields = [
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

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    best_balanced_acc = -1.0

    # 5. 라운드별 학습 및 평가 루프
    for r in range(1, args.rounds + 1):
        if args.method == "centralized":
            prev_model = copy.deepcopy(global_model)
            global_model, train_loss = train_one_client(
                model=global_model,
                dataloader=centralized_loader,
                criterion=criterion,
                lr=args.lr,
                local_epochs=args.local_epochs,
                device=device,
                mu=0.0,
                global_model=None,
            )
            avg_divergence = compute_weight_divergence(prev_model, global_model)
        else:
            effective_mu = args.mu if args.method == "fedprox" else 0.0
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

            # 샘플 수 비례 가중 평균으로 글로벌 모델 업데이트 및 평균 손실 계산
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
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
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
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row_data)
            f.flush()

        # 최고 balanced_acc 달성 시 .pt 가중치 저장
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(
                {
                    "round": r,
                    "method": args.method,
                    "mode": args.mode,
                    "mu": effective_mu,
                    "seed": args.seed,
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

    print(
        f"[Done] 학습 완료. 최고 Balanced Acc: {best_balanced_acc:.4f} | 결과 파일: {csv_path}"
    )


if __name__ == "__main__":
    main()

