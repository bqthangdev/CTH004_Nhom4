"""Tham số và tên bộ dữ liệu đã đối chiếu với code BIPN."""

DATASETS = {
    "tmall": ["click", "collect", "cart", "buy"],
    "taobao": ["view", "cart", "buy"],
    "yelp": ["tip", "neg", "neutral", "pos"],
    "ml": ["neg", "neutral", "pos"],
    "retailrocket": ["view", "cart", "buy"],
}

MODELS = ("bipn", "weighted", "lightgcn", "learned")


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
        self.behavior_boost = 1.02


def behavior_sample_weights(counts, behaviors, boost, neg_count):
    """Trọng số hành vi trong BCE chuẩn hóa trung bình bằng 1, tránh phóng đại cực đoan."""
    del neg_count
    alphas = _alphas(behaviors, boost)
    active = [alpha if count > 0 else 0.0 for alpha, count in zip(alphas, counts)]
    n_active = sum(1 for count in counts if count > 0)
    alpha_sum = sum(active)
    if n_active == 0 or alpha_sum == 0:
        raise ValueError("Không có mẫu BCE để tính trọng số hành vi.")
    return [alpha * n_active / alpha_sum for alpha in active]


def _alphas(behaviors, boost):
    target = behaviors[-1]
    weights = []
    for name in behaviors:
        weights.append(float(boost) if name == target or name == "cart" else 1.0)
    return weights
