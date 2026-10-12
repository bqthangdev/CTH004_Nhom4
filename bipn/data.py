"""Đọc tương tác đa hành vi và dựng đồ thị cho BIPN."""

import json
from pathlib import Path

import numpy as np


class Graph:
    def __init__(self, indices, values, n_nodes):
        self.indices = indices
        self.values = values
        self.n_nodes = n_nodes

    @property
    def n_edges(self):
        return 0 if self.indices is None else int(self.indices.shape[1])


class BehaviorData:
    """ID user và item bắt đầu từ 1. Chỉ số 0 dành cho padding."""

    def __init__(self, n_users, n_items, behaviors, train, test, valid):
        self.n_users = n_users
        self.n_items = n_items
        self.behaviors = list(behaviors)
        self.target = self.behaviors[-1]
        self.train = train
        self.test = test
        self.valid = valid
        self.all_items = _union_items(train, self.behaviors)
        self.all_graph = _bipartite_graph(n_users, n_items, _pairs_from(train, self.behaviors))
        self.target_graph = _bipartite_graph(n_users, n_items, _pairs_from(train, [self.target]))

    def bce_rows(self, rng, neg_count):
        rows = []
        for behavior_id, behavior in enumerate(self.behaviors):
            for user, items in self.train[behavior].items():
                blocked = self.all_items.get(user, set())
                for item in items:
                    rows.append((user, item, behavior_id, 1.0))
                    for _ in range(neg_count):
                        neg = _draw_negative(rng, self.n_items, blocked)
                        rows.append((user, neg, behavior_id, 0.0))
        return rows

    def bpr_rows(self, rng):
        rows = []
        for user, items in self.train[self.target].items():
            if not items:
                continue
            blocked = self.all_items.get(user, set())
            pos = int(rng.choice(items))
            neg = _draw_negative(rng, self.n_items, blocked)
            rows.append((user, pos, neg))
        return rows


def load_dataset(folder, behaviors):
    folder = Path(folder)
    count_path = folder / "count.txt"
    if not count_path.is_file():
        raise FileNotFoundError(f"Thiếu {count_path}. Cần count.txt với khóa user và item.")
    with count_path.open(encoding="utf-8") as handle:
        count = json.load(handle)
    n_users = int(count["user"])
    n_items = int(count["item"])
    train = {}
    for behavior in behaviors:
        path = folder / f"{behavior}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}")
        train[behavior] = _group_pairs(_read_pairs(path, n_users, n_items))
    test = _group_pairs(_read_pairs(folder / "test.txt", n_users, n_items))
    valid = _group_pairs(_read_pairs(folder / "validation.txt", n_users, n_items))
    return BehaviorData(n_users, n_items, behaviors, train, test, valid)


def make_synthetic(seed=2021):
    """Bộ nhỏ để kiểm tra khung chạy được, không phải số liệu của bài báo."""
    rng = np.random.default_rng(seed)
    n_users, n_items = 40, 25
    behaviors = ["view", "cart", "buy"]
    train = {name: {} for name in behaviors}
    test = {}
    valid = {}
    for user in range(1, n_users + 1):
        items = rng.choice(np.arange(1, n_items + 1), size=6, replace=False)
        view, cart, buy = items[:4], items[2:5], items[4:]
        train["view"][user] = [int(x) for x in view]
        train["cart"][user] = [int(x) for x in cart]
        held = int(buy[-1])
        train_buy = [int(x) for x in buy[:-1]]
        train["buy"][user] = train_buy
        if user <= 8:
            valid[user] = [held]
        else:
            test[user] = [held]
    return BehaviorData(n_users, n_items, behaviors, train, test, valid)


def _read_pairs(path, n_users, n_items):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}")
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no} cần đúng 2 cột user item")
            user, item = int(parts[0]), int(parts[1])
            if not (1 <= user <= n_users and 1 <= item <= n_items):
                raise ValueError(
                    f"{path}:{line_no} có id ngoài phạm vi user 1..{n_users}, item 1..{n_items}"
                )
            pairs.append((user, item))
    return pairs


def _group_pairs(pairs):
    grouped = {}
    for user, item in pairs:
        grouped.setdefault(user, [])
        if item not in grouped[user]:
            grouped[user].append(item)
    return grouped


def _pairs_from(train, behaviors):
    seen = set()
    pairs = []
    for behavior in behaviors:
        for user, items in train[behavior].items():
            for item in items:
                key = (user, item)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


def _union_items(train, behaviors):
    merged = {}
    for behavior in behaviors:
        for user, items in train[behavior].items():
            merged.setdefault(user, set()).update(items)
    return merged


def _draw_negative(rng, n_items, blocked):
    if len(blocked) >= n_items:
        raise ValueError("User đã tương tác với mọi item, không còn mẫu âm.")
    while True:
        item = int(rng.integers(1, n_items + 1))
        if item not in blocked:
            return item


def _bipartite_graph(n_users, n_items, pairs):
    n_user_nodes = n_users + 1
    n_nodes = n_user_nodes + n_items + 1
    if not pairs:
        return Graph(None, None, n_nodes)
    users = np.array([user for user, _ in pairs], dtype=np.int64)
    items = np.array([item for _, item in pairs], dtype=np.int64)
    item_nodes = items + n_user_nodes
    degree = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(degree, users, 1.0)
    np.add.at(degree, item_nodes, 1.0)
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = np.power(degree[nonzero], -0.5)
    weight = inv[users] * inv[item_nodes]
    rows = np.concatenate([users, item_nodes])
    cols = np.concatenate([item_nodes, users])
    indices = np.vstack([rows, cols])
    values = np.concatenate([weight, weight]).astype(np.float32)
    return Graph(indices, values, n_nodes)
