"""BIPN theo mục 3.2 của bài SIGIR 2024, không chép nguyên file gốc."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Graph


class LightGCN(nn.Module):
    """Tổng các tầng với trọng số 1/(l+1), l bắt đầu từ 0."""

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
        adj = torch.sparse_coo_tensor(
            self.indices,
            self.values,
            (self.n_nodes, self.n_nodes),
            device=x.device,
            check_invariants=False,
        ).coalesce()
        for layer in range(1, self.n_layers + 1):
            current = torch.sparse.mm(adj, current)
            out = out + current / (layer + 1)
        return out


class ItemPreference(nn.Module):
    """Ba lớp lọc trong công thức (4) đến (8)."""

    def __init__(self, embedding_size, n_behaviors):
        super().__init__()
        width = 2 * embedding_size + n_behaviors
        self.pre_filter = nn.Linear(width, embedding_size)
        self.item_layer = nn.Linear(width, embedding_size)
        self.post_filter = nn.Linear(width, embedding_size)

    def forward(self, user, item, behavior):
        raw = torch.cat((user, item, behavior), dim=-1)
        gate = torch.sigmoid(self.pre_filter(raw))
        filtered_user = gate * user
        aware = torch.tanh(self.item_layer(torch.cat((filtered_user, item, behavior), dim=-1)))
        post = torch.sigmoid(self.post_filter(raw))
        return post * aware


class BIPN(nn.Module):
    def __init__(self, n_users, n_items, behaviors, all_graph, target_graph, config):
        super().__init__()
        self.behaviors = list(behaviors)
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_nodes = n_users + 1
        dim = config.embedding_size
        self.user_embedding = nn.Embedding(self.n_user_nodes, dim)
        self.item_embedding = nn.Embedding(n_items + 1, dim)
        self.behavior_embedding = nn.Parameter(torch.eye(len(self.behaviors)))
        self.all_gcn = LightGCN(all_graph, config.n_layers)
        self.target_gcn = LightGCN(target_graph, config.n_layers)
        self.preference = ItemPreference(dim, len(self.behaviors))
        self.beta = config.beta
        self.gamma = config.gamma
        self.bce = nn.BCEWithLogitsLoss()
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def encode(self):
        initial = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        pretrained = self.all_gcn(initial)
        enhanced = self.target_gcn(pretrained)
        user_e, item_e = self._split(pretrained)
        user_hat, item_hat = self._split(enhanced)
        return user_e, item_e, user_hat, item_hat

    def loss(self, batch, target_items):
        user_e, item_e, user_hat, item_hat = self.encode()
        bce = self._bce_loss(batch["bce_user"], batch["bce_item"], batch["bce_behavior"], batch["bce_label"], user_e, item_e)
        bpr = self._bpr_loss(batch["bpr_user"], batch["bpr_pos"], batch["bpr_neg"], user_e, item_e, user_hat, item_hat, target_items)
        reg = self.user_embedding.weight.pow(2).sum() + self.item_embedding.weight.pow(2).sum()
        return self.beta * bce + (1.0 - self.beta) * bpr + self.gamma * reg

    def rank(self, users, target_items):
        user_e, item_e, user_hat, item_hat = self.encode()
        users = users.long()
        aggregated, lam = self._aggregate(users, user_e, item_e, target_items)
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
        return self.bce(logits, labels.float())

    def _bpr_loss(self, users, pos_items, neg_items, user_e, item_e, user_hat, item_hat, target_items):
        if users.numel() == 0:
            return user_e.new_zeros(())
        aggregated, lam = self._aggregate(users, user_e, item_e, target_items)
        pos = self._pair_score(users, pos_items, aggregated, lam, item_e, user_hat, item_hat)
        neg = self._pair_score(users, neg_items, aggregated, lam, item_e, user_hat, item_hat)
        return -F.logsigmoid(pos - neg).mean()

    def _pair_score(self, users, items, aggregated, lam, item_e, user_hat, item_hat):
        point = (aggregated * item_e[items]).sum(dim=-1)
        neighbor = (user_hat[users] * item_hat[items]).sum(dim=-1)
        return (1.0 - lam.squeeze(-1)) * point + lam.squeeze(-1) * neighbor

    def _aggregate(self, users, user_e, item_e, target_items):
        device = user_e.device
        dim = user_e.shape[1]
        target_id = len(self.behaviors) - 1
        behavior = self.behavior_embedding[target_id]
        aggregated = user_e.new_zeros((users.shape[0], dim))
        counts = []
        for row, user in enumerate(users.tolist()):
            items = target_items.get(int(user), [])
            counts.append(len(items))
            if not items:
                continue
            item_index = torch.tensor(items, dtype=torch.long, device=device)
            user_vec = user_e[int(user)].expand(item_index.shape[0], -1)
            behavior_vec = behavior.expand(item_index.shape[0], -1)
            pref = self.preference(user_vec, item_e[item_index], behavior_vec)
            aggregated[row] = pref.sum(dim=0)
        count = torch.tensor(counts, dtype=user_e.dtype, device=device).unsqueeze(-1)
        lam = torch.where(count > 0, 1.0 / count, torch.ones_like(count))
        return aggregated, lam

    def _split(self, nodes):
        return nodes[: self.n_user_nodes], nodes[self.n_user_nodes :]
