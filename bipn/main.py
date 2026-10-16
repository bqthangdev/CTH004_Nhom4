"""Điểm vào khung BIPN. Chạy --smoke để kiểm tra trên dữ liệu giả."""

import argparse
import random

import numpy as np
import torch

from checkpoint import is_better, read_checkpoint, save_checkpoint
from config import DATASETS, MODELS, Config, behavior_sample_weights
from data import load_dataset, make_synthetic
from model import BIPN, TargetLightGCN
from train import evaluate, train_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Khung BIPN")
    parser.add_argument("--smoke", action="store_true", help="Chạy vài bước trên dữ liệu giả")
    parser.add_argument("--data", default="", help="Thư mục đã có count.txt và các file hành vi")
    parser.add_argument("--dataset", default="tmall", choices=sorted(DATASETS))
    parser.add_argument("--model", default="bipn", choices=MODELS)
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="Nạp mô hình đã lưu và chấm lại")
    args = parser.parse_args()

    config = Config()
    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda"
    if args.eval and args.smoke:
        raise SystemExit("--eval không dùng cùng --smoke.")
    if args.eval:
        _check(args, config)
        return
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
    model = _build_model(args.model, data, config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    best = None
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, data, optimizer, config, rng)
        valid = evaluate(model, data, data.valid, config)
        test = evaluate(model, data, data.test, config)
        print(f"epoch {epoch} loss {loss:.4f} valid {valid} test {test}")
        if args.smoke or not is_better(valid, best):
            continue
        path = save_checkpoint(args.dataset, args.model, model, epoch, valid, test, config)
        best = valid
        print(f"save epoch {epoch} {path.name}")


def _check(args, config):
    if not args.data:
        raise SystemExit("Cần --data trỏ tới thư mục dữ liệu.")
    saved = read_checkpoint(args.dataset, args.model)
    if saved["dataset"] != args.dataset or saved["model"] != args.model:
        raise SystemExit("File mô hình không khớp --dataset và --model.")
    config.embedding_size = saved["embedding_size"]
    config.n_layers = saved["n_layers"]
    data = load_dataset(args.data, DATASETS[args.dataset])
    model = _build_model(args.model, data, config).to(config.device)
    model.load_state_dict(saved["state"])
    valid = evaluate(model, data, data.valid, config)
    test = evaluate(model, data, data.test, config)
    print(f"checkpoint {args.dataset} {args.model} epoch {saved['epoch']}")
    print(f"saved valid {saved['valid']} test {saved['test']}")
    print(f"reload valid {valid} test {test}")


def _build_model(name, data, config):
    if name == "lightgcn":
        return TargetLightGCN(data.n_users, data.n_items, data.target_graph, config)
    weights = None
    if name == "weighted":
        weights = behavior_sample_weights(
            data.positive_counts(), data.behaviors, config.behavior_boost, config.neg_count
        )
    return BIPN(
        data.n_users,
        data.n_items,
        data.behaviors,
        data.all_graph,
        data.target_graph,
        config,
        weights,
    )


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
