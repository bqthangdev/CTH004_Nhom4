"""Lưu epoch có HR@10 validation cao nhất, hòa thì lấy NDCG@10 cao hơn."""

from pathlib import Path

import torch

DIR = Path(__file__).resolve().parent / "checkpoints"


def file_of(dataset, model_name):
    return DIR / f"{dataset}_{model_name}.pt"


def is_better(valid, previous):
    if "HR@10" not in valid or "NDCG@10" not in valid:
        raise SystemExit("Cần HR@10 và NDCG@10 trong topk để chọn epoch lưu mô hình.")
    if previous is None:
        return True
    if valid["HR@10"] != previous["HR@10"]:
        return valid["HR@10"] > previous["HR@10"]
    return valid["NDCG@10"] > previous["NDCG@10"]


def save_checkpoint(dataset, model_name, model, epoch, valid, test, config):
    DIR.mkdir(exist_ok=True)
    path = file_of(dataset, model_name)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "dataset": dataset,
            "model": model_name,
            "epoch": epoch,
            "valid": valid,
            "test": test,
            "embedding_size": config.embedding_size,
            "n_layers": config.n_layers,
            "state": state,
        },
        path,
    )
    return path


def read_checkpoint(dataset, model_name):
    path = file_of(dataset, model_name)
    if not path.is_file():
        raise SystemExit(f"Chưa có {path.name}. Chạy train của {dataset} {model_name} trước.")
    return torch.load(path, map_location="cpu", weights_only=False)
