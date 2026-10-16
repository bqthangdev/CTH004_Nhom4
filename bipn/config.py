"""Tham số và tên bộ dữ liệu đã đối chiếu với code BIPN."""

DATASETS = {
    "tmall": ["click", "collect", "cart", "buy"],
    "taobao": ["view", "cart", "buy"],
    "yelp": ["tip", "neg", "neutral", "pos"],
    "ml": ["neg", "neutral", "pos"],
    "retailrocket": ["view", "cart", "buy"],
}

MODELS = ("bipn", "weighted", "lightgcn")


class Config:
    def __init__(self):
        self.embedding_size = 64
        self.n_layers = 2
        self.beta = 0.3
        self.gamma = 1e-4
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.batch_size = 1024
        self.neg_count = 4
        self.epochs = 200
        self.topk = (10, 20)
        self.seed = 2021
        self.device = "cpu"
        self.behavior_boost = 2.0


def behavior_sample_weights(counts, behaviors, boost, neg_count):
    """Trọng số từng dòng BCE để phần đóng góp kỳ vọng tỉ lệ với alpha."""
    alphas = _alphas(behaviors, boost)
    row_counts = [count * (1 + neg_count) for count in counts]
    total = sum(row_counts)
    alpha_sum = sum(alphas)
    if total == 0 or alpha_sum == 0:
        raise ValueError("Không có mẫu BCE để tính trọng số hành vi.")
    weights = []
    for rows, alpha in zip(row_counts, alphas):
        weights.append(0.0 if rows == 0 else alpha * total / (rows * alpha_sum))
    return weights


def _alphas(behaviors, boost):
    target = behaviors[-1]
    weights = []
    for name in behaviors:
        weights.append(float(boost) if name == target or name == "cart" else 1.0)
    return weights
