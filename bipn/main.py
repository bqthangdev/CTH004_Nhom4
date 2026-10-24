"""Điểm vào khung BIPN. Chạy --smoke để kiểm tra trên dữ liệu giả."""

import argparse
import random

import numpy as np
import torch

from checkpoint import (
    check_resume_matches,
    is_better,
    read_checkpoint,
    read_resume,
    restore_resume,
    resume_file_of,
    save_checkpoint,
    save_resume,
)
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
    parser.add_argument("--patience", type=int, default=5, help="Dừng sớm sau N epoch không tăng (0 để tắt)")
    parser.add_argument("--batch_size", type=int, default=0, help="Ghi đè batch_size (0 dùng mặc định 1024)")
    parser.add_argument("--eval", action="store_true", help="Nạp mô hình đã lưu và chấm lại")
    parser.add_argument("--resume", action="store_true", help="Chạy tiếp từ file .resume.pt của lần train trước")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)

    config = Config()
    if args.batch_size > 0:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda"
    if args.eval and args.smoke:
        raise SystemExit("--eval không dùng cùng --smoke.")
    if args.resume and args.smoke:
        raise SystemExit("--resume không dùng cùng --smoke.")
    if args.resume and args.eval:
        raise SystemExit("--resume không dùng cùng --eval.")
    if args.eval:
        _check(args, config)
        return
    saved = None
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
        saved = _resume_state(args, config)
        data = load_dataset(args.data, DATASETS[args.dataset])

    _seed(config.seed)
    model = _build_model(args.model, data, config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    best = None
    best_test = None
    best_epoch = 0
    start_epoch = 1
    patience = args.patience if not args.smoke else 0
    if args.resume:
        restore_resume(saved, model, optimizer, rng, config.device)
        best = saved["best_valid"]
        best_test = saved["best_test"]
        best_epoch = saved["best_epoch"]
        start_epoch = saved["epoch"] + 1
        print(f"resume from epoch {saved['epoch']}, best epoch {best_epoch}", flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        loss = train_one_epoch(model, data, optimizer, config, rng)
        valid = evaluate(model, data, data.valid, config)
        test = evaluate(model, data, data.test, config)
        line = f"epoch {epoch} loss {loss:.4f} valid {valid} test {test}"
        if args.smoke:
            print(line, flush=True)
            continue
        improved = is_better(valid, best)
        if improved:
            path = save_checkpoint(args.dataset, args.model, model, epoch, valid, test, config)
            best = valid
            best_test = test
            best_epoch = epoch
        stop = not improved and patience > 0 and best_epoch > 0 and (epoch - best_epoch) >= patience
        # Ghi resume sau mỗi epoch, sau khi best đã cập nhật. Ghi xong mới in, nên thấy dòng epoch là Ctrl+C được.
        save_resume(args.dataset, args.model, model, optimizer, rng, epoch, best_epoch, best, best_test, config)
        print(line, flush=True)
        if improved:
            print(f"save epoch {epoch} {path.name}", flush=True)
        if stop:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}", flush=True)
            break


def _resume_state(args, config):
    """Đọc file resume khi có --resume. Không có cờ thì từ chối nếu file đã tồn tại."""
    path = resume_file_of(args.dataset, args.model)
    if not args.resume:
        if path.is_file():
            raise SystemExit(
                f"Đã có {path.name} của lần train trước. Dùng --resume để chạy tiếp, "
                "hoặc xóa file đó nếu muốn train lại từ đầu."
            )
        return None
    saved = read_resume(args.dataset, args.model)
    check_resume_matches(saved, config, args.dataset, args.model)
    if saved["epoch"] >= config.epochs:
        raise SystemExit(
            f"{path.name} đã xong epoch {saved['epoch']}, hết số epoch {config.epochs}. "
            "Không học thêm. Tăng --epochs nếu muốn chạy tiếp."
        )
    if args.patience > 0 and saved["best_epoch"] > 0 and (saved["epoch"] - saved["best_epoch"]) >= args.patience:
        raise SystemExit(
            f"{path.name} đã tới ngưỡng dừng sớm: epoch {saved['epoch']}, best epoch {saved['best_epoch']}, "
            f"patience {args.patience}. Tăng --patience nếu muốn chạy tiếp."
        )
    return saved


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
    row_counts = None
    if name == "learned":
        row_counts = [count * (1 + config.neg_count) for count in data.positive_counts()]
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
        row_counts,
    )


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
