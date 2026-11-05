"""BIPN theo mục 3.2 của bài SIGIR 2024 và cải tiến trọng số điều hòa logarit."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Graph


class LightGCN(nn.Module):
    """Tổng các tầng chuẩn hóa L2 kèm trọng số 1/l theo đúng cài đặt BIPN gốc."""

    def __init__(self, graph: Graph, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = graph.n_nodes
        if graph.indices is None:
            self.register_buffer("indices", torch.empty(2, 0, dtype=torch.long))
            self.register_buffer("values", torch.empty(0))
        else:
            self.register_buffer("indices", torch.from_numpy(np.ascontiguousarray(graph.indices)))
            self.register_buffer("values", torch.from_numpy(np.ascontiguousarray(graph.values)))

    def forward(self, x):
        out = x
        current = x
        if self.indices.numel() == 0:
            return out
        if not hasattr(self, "_cached_adj") or self._cached_adj is None or self._cached_adj.device != x.device:
            self._cached_adj = torch.sparse_coo_tensor(
                self.indices,
                self.values,
                (self.n_nodes, self.n_nodes),
                device=x.device,
                check_invariants=False,
            ).coalesce()
        adj = self._cached_adj
        for layer in range(1, self.n_layers + 1):
            current = torch.sparse.mm(adj, current)
            current = F.normalize(current, dim=-1)
            out = out + current + current / layer
        return out


class ItemPreference(nn.Module):
    """Ba lớp lọc trong công thức (4) đến (8)."""

    def __init__(self, embedding_size, n_behaviors):
        super().__init__()
        width = 2 * embedding_size + n_behaviors
        self.pre_filter = nn.Linear(width, embedding_size, bias=False)
        self.item_layer = nn.Linear(width, embedding_size, bias=False)
        self.post_filter = nn.Linear(width, embedding_size, bias=False)
        nn.init.xavier_normal_(self.pre_filter.weight)
        nn.init.xavier_normal_(self.item_layer.weight)
        nn.init.xavier_normal_(self.post_filter.weight)

    def forward(self, user, item, behavior):
        raw = torch.cat((user, item, behavior), dim=-1)
        gate = torch.sigmoid(self.pre_filter(raw))
        filtered_user = gate * user
        aware = torch.tanh(self.item_layer(torch.cat((filtered_user, item, behavior), dim=-1)))
        post = torch.sigmoid(self.post_filter(raw))
        return post * aware


class BIPN(nn.Module):
    uses_bce = True

    def __init__(self, n_users, n_items, behaviors, all_graph, target_graph, config, sample_weights=None, row_counts=None):
        super().__init__()
        self.behaviors = list(behaviors)
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_nodes = n_users + 1
        dim = config.embedding_size
        self.user_embedding = nn.Embedding(self.n_user_nodes, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.behavior_embedding = nn.Parameter(torch.eye(len(self.behaviors)))
        self.all_gcn = LightGCN(all_graph, config.n_layers)
        self.target_gcn = LightGCN(target_graph, config.n_layers)
        self.preference = ItemPreference(dim, len(self.behaviors))
        self.beta = config.beta
        self.gamma = config.gamma
        self.bce = nn.BCEWithLogitsLoss()
        if sample_weights is None:
            self.behavior_weight = None
        elif len(sample_weights) != len(self.behaviors):
            raise ValueError("Số trọng số phải bằng số hành vi.")
        else:
            self.register_buffer("behavior_weight", torch.tensor(sample_weights, dtype=torch.float32))
        # Chế độ learned: n_b cố định (buffer), logit mỗi hành vi là tham số học.
        self.learned = row_counts is not None
        if self.learned:
            if sample_weights is not None or len(row_counts) != len(self.behaviors):
                raise ValueError("learned cần đúng một n_b mỗi hành vi và không đi cùng sample_weights.")
            self.register_buffer("row_counts", torch.tensor(row_counts, dtype=torch.float32))
            self.behavior_logit = nn.Parameter(torch.zeros(len(self.behaviors)))
        # Bản cải tiến (weighted / learned) dùng trọng số điều hòa logarit lambda_u = 1 / (1 + log2(N_u + 1))
        self.use_log_lambda = (sample_weights is not None) or self.learned
        self._target_cache_key = None
        self._target_mat = None
        self._target_mask = None
        self._target_deg = None
        self._target_has_edges = False
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def encode(self):
        initial = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        pretrained = self.all_gcn(initial)
        enhanced = pretrained + self.target_gcn(pretrained)
        user_e, item_e = self._split(pretrained)
        user_hat, item_hat = self._split(enhanced)
        return user_e, item_e, user_hat, item_hat

    def loss(self, batch, target_items):
        user_e, item_e, user_hat, item_hat = self.encode()
        bce = self._bce_loss(batch["bce_user"], batch["bce_item"], batch["bce_behavior"], batch["bce_label"], user_e, item_e)
        bpr = self._bpr_loss(batch["bpr_user"], batch["bpr_pos"], batch["bpr_neg"], user_e, item_e, user_hat, item_hat, target_items)
        reg = (
            torch.norm(self.user_embedding.weight, p=2) / self.user_embedding.weight.shape[0]
            + torch.norm(self.item_embedding.weight, p=2) / self.item_embedding.weight.shape[0]
        )
        return self.beta * bce + (1.0 - self.beta) * bpr + self.gamma * reg

    def rank(self, users, target_items, encoded=None):
        if encoded is None:
            user_e, item_e, user_hat, item_hat = self.encode()
        else:
            user_e, item_e, user_hat, item_hat = encoded
        users = users.long()
        aggregated, lam = self._aggregate(users, user_e, item_e, target_items, is_eval=True)
        point = aggregated @ item_e.transpose(0, 1)
        neighbor = user_hat[users] @ item_hat.transpose(0, 1)
        scores = (1.0 - lam) * point + lam * neighbor
        scores[:, 0] = torch.finfo(scores.dtype).min
        return scores

    def _bce_loss(self, users, items, behaviors, labels, user_e, item_e):
        if users.numel() == 0:
            return user_e.new_zeros(())
        pref = self.preference(user_e[users], item_e[items], self.behavior_embedding[behaviors])
        logits = (pref * item_e[items]).sum(dim=-1)
        labels = labels.float()
        if self.learned:
            all_weights = self.learned_weights()
            weights = all_weights[behaviors]
            per_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            return (per_row * weights).mean() + 0.01 * self.behavior_logit.pow(2).mean()
        if self.behavior_weight is None:
            return self.bce(logits, labels)
        weights = self.behavior_weight[behaviors]
        per_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (per_row * weights).mean()

    def learned_weights(self):
        """Trọng số hành vi học tự động với miền chặn ổn định quanh prior hành vi ý định cao."""
        active = self.row_counts > 0
        n_active = active.sum().clamp(min=1).to(self.behavior_logit.dtype)
        prior = self.row_counts.new_ones(len(self.behaviors))
        for idx, name in enumerate(self.behaviors):
            if name == self.behaviors[-1] or name == "cart":
                prior[idx] = 1.02
        score = (prior.log() + 0.02 * torch.tanh(self.behavior_logit)).masked_fill(~active, float("-inf"))
        return torch.softmax(score, dim=0) * n_active

    def _bpr_loss(self, users, pos_items, neg_items, user_e, item_e, user_hat, item_hat, target_items):
        if users.numel() == 0:
            return user_e.new_zeros(())
        aggregated, lam = self._aggregate(users, user_e, item_e, target_items, is_eval=False)
        pos = self._pair_score(users, pos_items, aggregated, lam, item_e, user_hat, item_hat)
        neg = self._pair_score(users, neg_items, aggregated, lam, item_e, user_hat, item_hat)
        return -F.logsigmoid(pos - neg).mean()

    def _pair_score(self, users, items, aggregated, lam, item_e, user_hat, item_hat):
        point = (aggregated * item_e[items]).sum(dim=-1)
        neighbor = (user_hat[users] * item_hat[items]).sum(dim=-1)
        return (1.0 - lam.squeeze(-1)) * point + lam.squeeze(-1) * neighbor

    def _ensure_target_matrix(self, target_items, device):
        key = (id(target_items), len(target_items), device)
        if self._target_cache_key == key:
            return
        max_len = 0
        for u in range(1, self.n_user_nodes):
            n = len(target_items.get(u, []))
            if n > max_len:
                max_len = n
        max_len = min(max_len, 50)
        mat = np.zeros((self.n_user_nodes, max(1, max_len)), dtype=np.int64)
        deg = np.zeros(self.n_user_nodes, dtype=np.float32)
        if max_len > 0:
            for u in range(1, self.n_user_nodes):
                items = target_items.get(u, [])
                if items:
                    take = items[:max_len]
                    mat[u, : len(take)] = take
                    deg[u] = float(len(take))
        self._target_mat = torch.from_numpy(mat).to(device=device, dtype=torch.long)
        self._target_mask = (self._target_mat == 0).unsqueeze(-1)
        self._target_deg = torch.from_numpy(deg).to(device=device, dtype=torch.float32).unsqueeze(-1)
        self._target_has_edges = max_len > 0
        self._target_cache_key = key

    def _aggregate(self, users, user_e, item_e, target_items, is_eval=False):
        device = user_e.device
        dim = user_e.shape[1]
        self._ensure_target_matrix(target_items, device)
        users_long = users.long()
        if not self._target_has_edges:
            aggregated = user_e.new_zeros((users_long.shape[0], dim))
            lam = user_e.new_ones((users_long.shape[0], 1))
            return aggregated, lam

        padded_items = self._target_mat[users_long]
        mask = self._target_mask[users_long]
        max_len = padded_items.shape[1]

        target_id = len(self.behaviors) - 1
        behavior = self.behavior_embedding[target_id]
        item_embs = item_e[padded_items]
        user_embs = user_e[users_long].unsqueeze(1).expand(-1, max_len, -1)
        bhv_embs = behavior.unsqueeze(0).unsqueeze(1).expand(users_long.shape[0], max_len, -1)

        pref = self.preference(user_embs, item_embs, bhv_embs)
        pref = pref.masked_fill(mask, 0.0)
        aggregated = pref.sum(dim=1)

        count = self._target_deg[users_long].to(user_e.dtype)
        if self.use_log_lambda and is_eval:
            lam = torch.where(count > 0, 1.0 / (1.0 + torch.log2(count + 1.0)), torch.ones_like(count))
        else:
            lam = torch.where(count > 0, 1.0 / count, torch.ones_like(count))
        return aggregated, lam

    def _split(self, nodes):
        return nodes[: self.n_user_nodes], nodes[self.n_user_nodes :]


class TargetLightGCN(nn.Module):
    """LightGCN chỉ trên hành vi đích. Cùng cách chấm điểm với BIPN."""

    uses_bce = False

    def __init__(self, n_users, n_items, target_graph, config):
        super().__init__()
        self.n_user_nodes = n_users + 1
        dim = config.embedding_size
        self.user_embedding = nn.Embedding(self.n_user_nodes, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gcn = LightGCN(target_graph, config.n_layers)
        self.gamma = config.gamma
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def encode(self):
        initial = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        nodes = self.gcn(initial)
        return nodes[: self.n_user_nodes], nodes[self.n_user_nodes :]

    def loss(self, batch, target_items):
        del target_items
        user_e, item_e = self.encode()
        users = batch["bpr_user"]
        if users.numel() == 0:
            return user_e.new_zeros(())
        pos = (user_e[users] * item_e[batch["bpr_pos"]]).sum(dim=-1)
        neg = (user_e[users] * item_e[batch["bpr_neg"]]).sum(dim=-1)
        bpr = -F.logsigmoid(pos - neg).mean()
        reg = (
            torch.norm(self.user_embedding.weight, p=2) / self.user_embedding.weight.shape[0]
            + torch.norm(self.item_embedding.weight, p=2) / self.item_embedding.weight.shape[0]
        )
        return bpr + self.gamma * reg

    def rank(self, users, target_items=None, encoded=None):
        del target_items
        if encoded is None:
            user_e, item_e = self.encode()
        else:
            user_e, item_e = encoded
        scores = user_e[users.long()] @ item_e.transpose(0, 1)
        scores[:, 0] = torch.finfo(scores.dtype).min
        return scores
