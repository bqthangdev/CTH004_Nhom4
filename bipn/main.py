"""Điểm vào khung BIPN. Chạy --smoke để kiểm tra trên dữ liệu giả."""

import argparse
import random

import numpy as np
import torch

from config import DATASETS, Config
from data import load_dataset, make_synthetic
from model import BIPN
from train import evaluate, train_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Khung BIPN")
    parser.add_argument("--smoke", action="store_true", help="Chạy vài bước trên dữ liệu giả")
    parser.add_argument("--data", default="", help="Thư mục đã có count.txt và các file hành vi")
    parser.add_argument("--dataset", default="tmall", choices=sorted(DATASETS))
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    config = Config()
    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda"
    if args.smoke:
        config.embedding_size = 8
        config.n_layers = 1
        config.batch_size = 32
        config.neg_count = 1
        config.epochs = 2
        config.device = "cpu"
        data = make_synthetic(config.seed)
    else:
        if not args.data:
            raise SystemExit("Cần --data trỏ tới thư mục dữ liệu, hoặc dùng --smoke.")
        if args.epochs:
            config.epochs = args.epochs
        data = load_dataset(args.data, DATASETS[args.dataset])

    _seed(config.seed)
    model = BIPN(
        data.n_users,
        data.n_items,
        data.behaviors,
        data.all_graph,
        data.target_graph,
        config,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, data, optimizer, config, rng)
        valid = evaluate(model, data, data.valid, config)
        test = evaluate(model, data, data.test, config)
        print(f"epoch {epoch} loss {loss:.4f} valid {valid} test {test}")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
