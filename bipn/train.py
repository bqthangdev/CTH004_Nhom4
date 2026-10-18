"""Vòng huấn luyện và đánh giá xếp hạng."""

import numpy as np
import torch

from metrics import hit_ndcg


def train_one_epoch(model, data, optimizer, config, rng):
    model.train()
    bce_rows = data.bce_rows(rng, config.neg_count) if model.uses_bce else []
    bpr_rows = data.bpr_rows(rng)
    rng.shuffle(bce_rows)
    rng.shuffle(bpr_rows)
    losses = []
    n_rows = max(len(bce_rows), len(bpr_rows))
    n_batches = max(1, int(np.ceil(n_rows / config.batch_size))) if n_rows else 1
    for step in range(n_batches):
        start = step * config.batch_size
        bce_batch = _window(bce_rows, start, config.batch_size)
        bpr_batch = _window(bpr_rows, start, config.batch_size)
        batch = _to_batch(bce_batch, bpr_batch, config.device)
        optimizer.zero_grad()
        loss = model.loss(batch, data.train[data.target])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, data, split, config):
    model.eval()
    users = sorted(split)
    if not users:
        return {f"HR@{k}": 0.0 for k in config.topk} | {f"NDCG@{k}": 0.0 for k in config.topk}
    max_k = max(config.topk)
    relevance = []
    n_relevant = []
    train_target = data.train[data.target]
    encoded = model.encode()
    for start in range(0, len(users), config.batch_size):
        chunk = users[start : start + config.batch_size]
        index = torch.tensor(chunk, dtype=torch.long, device=config.device)
        scores = model.rank(index, train_target, encoded=encoded).cpu().numpy()
        for row, user in enumerate(chunk):
            seen = train_target.get(user, [])
            if seen:
                scores[row, seen] = -np.inf
            take = min(max_k, scores.shape[1])
            top = np.argpartition(-scores[row], take - 1)[:take]
            top = top[np.argsort(-scores[row, top])]
            truth = set(split[user])
            relevance.append([item in truth for item in top])
            n_relevant.append(len(truth))
    width = max(len(row) for row in relevance)
    padded = np.zeros((len(relevance), width), dtype=bool)
    for row, flags in enumerate(relevance):
        padded[row, : len(flags)] = flags
    usable = tuple(k for k in config.topk if k <= padded.shape[1])
    return hit_ndcg(padded, np.array(n_relevant), usable)


def _window(rows, start, size):
    if not rows:
        return []
    start = start % len(rows)
    chunk = rows[start : start + size]
    if len(chunk) < size and len(rows) >= size:
        chunk = chunk + rows[: size - len(chunk)]
    return chunk


def _to_batch(bce_rows, bpr_rows, device):
    def column(rows, index, dtype):
        if not rows:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.tensor([row[index] for row in rows], dtype=dtype, device=device)

    return {
        "bce_user": column(bce_rows, 0, torch.long),
        "bce_item": column(bce_rows, 1, torch.long),
        "bce_behavior": column(bce_rows, 2, torch.long),
        "bce_label": column(bce_rows, 3, torch.float),
        "bpr_user": column(bpr_rows, 0, torch.long),
        "bpr_pos": column(bpr_rows, 1, torch.long),
        "bpr_neg": column(bpr_rows, 2, torch.long),
    }
