"""Nạp các mô hình của pipeline và chấm lại validation, test."""

import subprocess
import sys
from pathlib import Path

import torch

from run_pipeline import STEPS, dataset_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chấm lại validation và test từ checkpoint")
    parser.add_argument("--dataset", choices=["all", "tmall", "retailrocket"], default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch không thấy GPU. Cài bản CUDA trước khi chạy pipeline.")
    steps = STEPS if args.dataset == "all" else [s for s in STEPS if s[0] == args.dataset]
    for dataset, model in steps:
        command = [
            sys.executable,
            "main.py",
            "--eval",
            "--data",
            str(dataset_dir(dataset)),
            "--dataset",
            dataset,
            "--model",
            model,
            "--device",
            "cuda",
        ]
        print(f"check {dataset} {model}", flush=True)
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent)
        if result.returncode != 0:
            raise SystemExit(f"{dataset} {model} không kiểm tra được.")


if __name__ == "__main__":
    main()
