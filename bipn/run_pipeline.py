"""Chạy đủ thí nghiệm trên GPU, đúng thứ tự bài tập."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from prepare_retailrocket import convert

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STEPS = (
    ("tmall", "weighted"),
    ("tmall", "lightgcn"),
    ("retailrocket", "bipn"),
    ("retailrocket", "weighted"),
    ("retailrocket", "lightgcn"),
)


def main():
    parser = argparse.ArgumentParser(description="Chạy BIPN, bản cải tiến và LightGCN trên hai bộ dữ liệu")
    parser.add_argument("--epochs", type=int, default=0, help="Ghi đè số epoch. Bỏ trống để dùng 200.")
    parser.add_argument("--patience", type=int, default=5, help="Dừng sớm sau N epoch không tăng (0 để tắt)")
    parser.add_argument("--batch_size", type=int, default=0, help="Ghi đè batch_size (0 dùng mặc định 1024)")
    parser.add_argument("--dataset", choices=["all", "tmall", "retailrocket"], default="all", help="Chỉ định bộ dữ liệu cần chạy (mặc định all)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch không thấy GPU. Cài bản CUDA trước khi chạy pipeline.")
    steps = STEPS if args.dataset == "all" else [s for s in STEPS if s[0] == args.dataset]
    _prepare(steps)
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    for dataset, model in steps:
        folder = dataset_dir(dataset)
        command = [
            sys.executable,
            "main.py",
            "--data",
            str(folder),
            "--dataset",
            dataset,
            "--model",
            model,
            "--device",
            "cuda",
        ]
        if args.epochs:
            command.extend(["--epochs", str(args.epochs)])
        if args.patience is not None:
            command.extend(["--patience", str(args.patience)])
        if args.batch_size:
            command.extend(["--batch_size", str(args.batch_size)])
        log_path = log_dir / f"{dataset}_{model}.log"
        print(f"start {dataset} {model} -> {log_path.name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parent, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(f"{dataset} {model} dừng. Xem {log_path}")
        print(f"done {dataset} {model}", flush=True)


def dataset_dir(name):
    if name == "tmall":
        return DATA / "Tmall"
    if name == "retailrocket":
        return DATA / "retailrocket"
    raise SystemExit(f"Pipeline không có thư mục cho {name}.")


def _prepare(steps):
    datasets = {s[0] for s in steps}
    if "tmall" in datasets:
        tmall = DATA / "Tmall"
        needed = ["count.txt", "click.txt", "collect.txt", "cart.txt", "buy.txt", "test.txt", "validation.txt"]
        missing = [name for name in needed if not (tmall / name).is_file()]
        if missing:
            raise SystemExit("Thiếu file Tmall: " + ", ".join(missing))
    if "retailrocket" in datasets:
        retailrocket = DATA / "retailrocket"
        if (retailrocket / "count.txt").is_file():
            return
        events = retailrocket / "events.csv"
        if not events.is_file():
            raise SystemExit("Thiếu data/retailrocket/events.csv")
        summary = convert(events, retailrocket)
        print(f"retailrocket users {summary['users']} items {summary['items']}", flush=True)


if __name__ == "__main__":
    main()
