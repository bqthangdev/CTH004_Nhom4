"""Lưu epoch có HR@10 validation cao nhất, hòa thì lấy NDCG@10 cao hơn.

File `{dataset}_{model}.pt` giữ mô hình tốt nhất. File `{dataset}_{model}.resume.pt`
giữ trạng thái sau epoch gần nhất để chạy tiếp bằng --resume.
"""

import os
import random
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parent / "checkpoints"

# Các trường phải trùng giữa file resume và lệnh --resume. --epochs và --patience được đổi.
RESUME_FIELDS = (
    "dataset",
    "model",
    "embedding_size",
    "n_layers",
    "batch_size",
    "lr",
    "weight_decay",
    "neg_count",
    "beta",
    "gamma",
    "seed",
)


def file_of(dataset, model_name):
    return DIR / f"{dataset}_{model_name}.pt"


def resume_file_of(dataset, model_name):
    return DIR / f"{dataset}_{model_name}.resume.pt"


def is_better(valid, previous):
    if "HR@10" not in valid or "NDCG@10" not in valid:
        raise SystemExit("Cần HR@10 và NDCG@10 trong topk để chọn epoch lưu mô hình.")
    if previous is None:
        return True
    if valid["HR@10"] != previous["HR@10"]:
        return valid["HR@10"] > previous["HR@10"]
    return valid["NDCG@10"] > previous["NDCG@10"]


def save_checkpoint(dataset, model_name, model, epoch, valid, test, config):
    path = file_of(dataset, model_name)
    _atomic_save(
        {
            "dataset": dataset,
            "model": model_name,
            "epoch": epoch,
            "valid": valid,
            "test": test,
            "embedding_size": config.embedding_size,
            "n_layers": config.n_layers,
            "state": _to_cpu(model.state_dict()),
        },
        path,
    )
    return path


def read_checkpoint(dataset, model_name):
    path = file_of(dataset, model_name)
    if not path.is_file():
        raise SystemExit(f"Chưa có {path.name}. Chạy train của {dataset} {model_name} trước.")
    return torch.load(path, map_location="cpu", weights_only=False)


def save_resume(dataset, model_name, model, optimizer, rng, epoch, best_epoch, best_valid, best_test, config):
    """Ghi trạng thái sau khi epoch đã chấm xong và best đã cập nhật."""
    path = resume_file_of(dataset, model_name)
    payload = {
        "dataset": dataset,
        "model": model_name,
        "embedding_size": config.embedding_size,
        "n_layers": config.n_layers,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "neg_count": config.neg_count,
        "beta": config.beta,
        "gamma": config.gamma,
        "seed": config.seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_test": best_test,
        "state": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
    }
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    _atomic_save(payload, path)
    return path


def read_resume(dataset, model_name):
    path = resume_file_of(dataset, model_name)
    if not path.is_file():
        raise SystemExit(f"Chưa có {path.name}. Bỏ --resume để train mới {dataset} {model_name}.")
    return torch.load(path, map_location="cpu", weights_only=False)


def check_resume_matches(saved, config, dataset, model_name):
    """Từ chối nếu file resume lệch cấu hình. Chỉ --epochs và --patience được đổi."""
    current = {"dataset": dataset, "model": model_name}
    for field in RESUME_FIELDS:
        if field not in current:
            current[field] = getattr(config, field)
        if saved.get(field) != current[field]:
            raise SystemExit(
                f"File resume lệch {field}: file ghi {saved.get(field)!r}, lệnh hiện tại {current[field]!r}. "
                "Khi --resume chỉ được đổi --epochs và --patience."
            )


def restore_resume(saved, model, optimizer, rng, device):
    """Nạp trọng số, Adam và RNG vào các đối tượng đã dựng sẵn."""
    model.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            # `step` của Adam nằm trên CPU khi không dùng capturable hay fused.
            if key != "step" and torch.is_tensor(value):
                state[key] = value.to(device)
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    if "cuda_rng" in saved and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def _atomic_save(payload, path):
    """Ghi ra .tmp cùng thư mục, fsync, rồi os.replace. Tắt máy lúc đang ghi thì bản trước còn."""
    DIR.mkdir(exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value
