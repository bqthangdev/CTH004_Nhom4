"""HR và NDCG theo cách code BIPN đang tính, không theo câu chữ docstring."""

import numpy as np


def hit_ndcg(relevance, n_relevant, ks):
    """relevance: bool [n_users, max_k], True nếu vị trí đó là item kiểm tra."""
    max_k = relevance.shape[1]
    result = {}
    for k in ks:
        if k > max_k:
            raise ValueError(f"topk {k} lớn hơn danh sách đã xếp {max_k}")
        hits = []
        gains = []
        for row in range(relevance.shape[0]):
            chosen = relevance[row, :k]
            hits.append(1.0 if chosen.any() else 0.0)
            discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1)
            dcg = float((chosen * discounts).sum())
            ideal_len = int(min(n_relevant[row], k))
            idcg = float((1.0 / np.log2(np.arange(1, ideal_len + 1) + 1)).sum()) if ideal_len else 0.0
            gains.append(dcg / idcg if idcg else 0.0)
        result[f"HR@{k}"] = float(np.mean(hits)) if hits else 0.0
        result[f"NDCG@{k}"] = float(np.mean(gains)) if gains else 0.0
    return result
