"""Đọc tương tác đa hành vi và dựng đồ thị cho BIPN."""

import json
from pathlib import Path

import numpy as np
import torch


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
        self._build_fast_structures()

    def _build_fast_structures(self):
        pos_u, pos_i, pos_b = [], [], []
        for behavior_id, behavior in enumerate(self.behaviors):
            for user, items in self.train[behavior].items():
                if items:
                    pos_u.extend([user] * len(items))
                    pos_i.extend(items)
                    pos_b.extend([behavior_id] * len(items))
        self.pos_users = np.array(pos_u, dtype=np.int64)
        self.pos_items = np.array(pos_i, dtype=np.int64)
        self.pos_behaviors = np.array(pos_b, dtype=np.int64)

        self.blocked_matrix = np.zeros((self.n_users + 1, self.n_items + 1), dtype=bool)
        self.blocked_matrix[:, 0] = True
        if self.pos_users.size > 0:
            self.blocked_matrix[self.pos_users, self.pos_items] = True
        self.all_blocked = np.all(self.blocked_matrix[:, 1:], axis=1)

        self.buy_counts = np.zeros(self.n_users + 1, dtype=np.int64)
        flat_buy = []
        target_dict = self.train[self.target]
        for user in range(1, self.n_users + 1):
            items = target_dict.get(user, [])
            self.buy_counts[user] = len(items)
            if items:
                flat_buy.extend(items)
        self.buy_offsets = np.zeros(self.n_users + 1, dtype=np.int64)
        np.cumsum(self.buy_counts[:-1], out=self.buy_offsets[1:])
        self.flat_buy_items = np.array(flat_buy, dtype=np.int64)

    def _sample_negatives(self, rng, users):
        if users.size == 0:
            return np.empty(0, dtype=np.int64)
        if np.any(self.all_blocked[users]):
            raise ValueError("User đã tương tác với mọi item, không còn mẫu âm.")
        items = rng.integers(1, self.n_items + 1, size=users.shape[0], dtype=np.int64)
        mask = self.blocked_matrix[users, items]
        while np.any(mask):
            bad_idx = np.flatnonzero(mask)
            items[bad_idx] = rng.integers(1, self.n_items + 1, size=bad_idx.shape[0], dtype=np.int64)
            mask = self.blocked_matrix[users, items]
        return items

    def iter_batches(self, rng, batch_size, neg_count, uses_bce, device):
        """Sinh các batch huấn luyện đồng bộ theo tương tác dương giống cài đặt BIPN gốc."""
        n_pos = self.pos_users.shape[0]
        if n_pos == 0:
            yield {
                "bce_user": torch.empty(0, dtype=torch.long, device=device),
                "bce_item": torch.empty(0, dtype=torch.long, device=device),
                "bce_behavior": torch.empty(0, dtype=torch.long, device=device),
                "bce_label": torch.empty(0, dtype=torch.float32, device=device),
                "bpr_user": torch.empty(0, dtype=torch.long, device=device),
                "bpr_pos": torch.empty(0, dtype=torch.long, device=device),
                "bpr_neg": torch.empty(0, dtype=torch.long, device=device),
            }
            return

        perm = rng.permutation(n_pos)
        for start in range(0, n_pos, batch_size):
            idx = perm[start : start + batch_size]
            b_users = self.pos_users[idx]
            b_items = self.pos_items[idx]
            b_bhvs = self.pos_behaviors[idx]

            if uses_bce:
                if neg_count > 0:
                    neg_u = np.repeat(b_users, neg_count)
                    neg_b = np.repeat(b_bhvs, neg_count)
                    neg_i = self._sample_negatives(rng, neg_u)
                    bce_u = np.concatenate([b_users, neg_u])
                    bce_i = np.concatenate([b_items, neg_i])
                    bce_b = np.concatenate([b_bhvs, neg_b])
                    bce_lbl = np.empty(bce_u.shape[0], dtype=np.float32)
                    bce_lbl[: b_users.shape[0]] = 1.0
                    bce_lbl[b_users.shape[0] :] = 0.0
                else:
                    bce_u = b_users
                    bce_i = b_items
                    bce_b = b_bhvs
                    bce_lbl = np.ones(b_users.shape[0], dtype=np.float32)
                bce_user_t = torch.from_numpy(bce_u).to(device=device, dtype=torch.long)
                bce_item_t = torch.from_numpy(bce_i).to(device=device, dtype=torch.long)
                bce_bhv_t = torch.from_numpy(bce_b).to(device=device, dtype=torch.long)
                bce_lbl_t = torch.from_numpy(bce_lbl).to(device=device, dtype=torch.float32)
            else:
                bce_user_t = torch.empty(0, dtype=torch.long, device=device)
                bce_item_t = torch.empty(0, dtype=torch.long, device=device)
                bce_bhv_t = torch.empty(0, dtype=torch.long, device=device)
                bce_lbl_t = torch.empty(0, dtype=torch.float32, device=device)

            counts = self.buy_counts[b_users]
            valid_mask = counts > 0
            bpr_u = b_users[valid_mask]
            if bpr_u.size > 0:
                u_counts = counts[valid_mask]
                u_offsets = self.buy_offsets[bpr_u]
                rand_idx = (rng.random(bpr_u.shape[0]) * u_counts).astype(np.int64)
                bpr_pos = self.flat_buy_items[u_offsets + rand_idx]
                bpr_neg = self._sample_negatives(rng, bpr_u)
                bpr_user_t = torch.from_numpy(bpr_u).to(device=device, dtype=torch.long)
                bpr_pos_t = torch.from_numpy(bpr_pos).to(device=device, dtype=torch.long)
                bpr_neg_t = torch.from_numpy(bpr_neg).to(device=device, dtype=torch.long)
            else:
                bpr_user_t = torch.empty(0, dtype=torch.long, device=device)
                bpr_pos_t = torch.empty(0, dtype=torch.long, device=device)
                bpr_neg_t = torch.empty(0, dtype=torch.long, device=device)

            yield {
                "bce_user": bce_user_t,
                "bce_item": bce_item_t,
                "bce_behavior": bce_bhv_t,
                "bce_label": bce_lbl_t,
                "bpr_user": bpr_user_t,
                "bpr_pos": bpr_pos_t,
                "bpr_neg": bpr_neg_t,
            }

    def positive_counts(self):
        return [sum(len(items) for items in self.train[name].values()) for name in self.behaviors]

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
