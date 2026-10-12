"""Tham số và tên bộ dữ liệu đã đối chiếu với code BIPN."""

DATASETS = {
    "tmall": ["click", "collect", "cart", "buy"],
    "taobao": ["view", "cart", "buy"],
    "yelp": ["tip", "neg", "neutral", "pos"],
    "ml": ["neg", "neutral", "pos"],
}


class Config:
    def __init__(self):
        self.embedding_size = 64
        self.n_layers = 2
        self.beta = 0.5
        self.gamma = 1e-4
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.batch_size = 1024
        self.neg_count = 4
        self.epochs = 200
        self.topk = (10, 20)
        self.seed = 2021
        self.device = "cpu"
