from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import math
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_dataset(data_id: int, data_root: str = '/tmp'):
    if data_id not in DATASET_NAMES:
        raise ValueError(f'dataset id must be 0..8, got {data_id}')
    try:
        from torch_geometric.datasets import Actor, WebKB, WikipediaNetwork
        from torch_geometric.datasets.heterophilous_graph_dataset import HeterophilousGraphDataset
    except ImportError as exc:
        raise ImportError('Dataset loading requires torch-geometric; model-only tests do not.') from exc
    roots = ('RomanEmpire', 'Minesweeper', 'AmazonRatings', 'Chameleon',
             'Squirrel', 'Actor', 'Cornell', 'Texas', 'Wisconsin')
    root = str(Path(data_root) / roots[data_id])
    if data_id <= 2:
        names = ('Roman-empire', 'Minesweeper', 'Amazon-ratings')
        dataset = HeterophilousGraphDataset(root=root, name=names[data_id])
    elif data_id in (3, 4):
        dataset = WikipediaNetwork(root=root, name=roots[data_id].lower(), geom_gcn_preprocess=True)
    elif data_id == 5:
        dataset = Actor(root=root)
    else:
        dataset = WebKB(root=root, name=roots[data_id])
    return dataset, dataset.num_classes


DATASET_NAMES = {
    0: 'Roman-empire',
    1: 'Minesweeper',
    2: 'Amazon-ratings',
    3: 'Chameleon',
    4: 'Squirrel',
    5: 'Actor',
    6: 'Cornell',
    7: 'Texas',
    8: 'Wisconsin',
}


def set_seed(seed: int = 0, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(deterministic)


def get_device(device_arg: str = 'auto') -> torch.device:
    if device_arg.lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_arg)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; refusing a silent device change.')
    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('MPS requested but not available.')
    return device


def pick_single_split(data, split_idx: int = 0):

    data = data.clone() if hasattr(data, 'clone') else copy.deepcopy(data)
    generator = torch.Generator().manual_seed(split_idx)
    perm = torch.randperm(data.num_nodes, generator=generator)
    train_end = int(data.num_nodes * 0.10)
    val_end = int(data.num_nodes * 0.55)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True
    return data


def unique_directed_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index

    row, col = edge_index
    key = row.to(torch.long) * int(num_nodes) + col.to(torch.long)
    perm = torch.argsort(key)

    row = row[perm]
    col = col[perm]
    key = key[perm]

    keep = torch.ones_like(key, dtype=torch.bool)
    keep[1:] = key[1:] != key[:-1]
    return torch.stack([row[keep], col[keep]], dim=0)


def prepare_graph(data):
    num_nodes = data.num_nodes

    edge_index = data.edge_index
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must be int64 with shape [2, m].')
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError('Edge endpoint outside the node range.')
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    edge_index = unique_directed_edges(
        torch.cat((edge_index, edge_index.flip(0)), dim=1), num_nodes=num_nodes)

    row, col = edge_index
    oriented_mask = row < col
    edge_index_solver = torch.stack([row[oriented_mask], col[oriented_mask]], dim=0)

    if edge_index_solver.numel() == 0:
        max_degree = 0
    else:
        deg = torch.bincount(
            torch.cat([edge_index_solver[0], edge_index_solver[1]], dim=0),
            minlength=num_nodes,
        )
        max_degree = int(deg.max().item())

    deg_mp = torch.bincount(col, minlength=num_nodes).to(torch.float32).clamp_min_(1.0)

    data.edge_index_mp = edge_index
    data.edge_index_solver = edge_index_solver
    data.deg_mp = deg_mp
    data.max_degree = max_degree
    return data


def inverse_softplus(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        raise ValueError('inverse_softplus requires a finite positive value.')
    return x + math.log(-math.expm1(-x))


class BoundedInfluencePotential(nn.Module):
    def __init__(self, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis
        self.raw_b0 = nn.Parameter(torch.tensor(-2.25))
        self.raw_a = nn.Parameter(torch.full((num_basis,), -5.0))
        self.raw_beta = nn.Parameter(torch.full((num_basis,), -1.0))
        self.c = nn.Parameter(torch.linspace(-3.0, 3.0, steps=num_basis))

    def constrained_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = F.softplus(self.raw_a)
        beta = F.softplus(self.raw_beta)
        b0 = F.softplus(self.raw_b0)
        c = self.c
        return a, beta, c, b0

    def psi(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, b0 = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        sig = torch.sigmoid(u)
        return b0 + (a * sig).sum(dim=-1)

    def psi_prime(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, _ = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        sig = torch.sigmoid(u)
        return (a * beta * sig * (1.0 - sig)).sum(dim=-1)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, b0 = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        base = F.softplus(c)
        return b0 * t + ((a / beta) * (F.softplus(u) - base)).sum(dim=-1)


class FixedPotential(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        if family not in ('tv', 'quadratic'):
            raise ValueError(f'Unknown fixed potential: {family}')
        self.family = family

    def psi(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t) if self.family == 'tv' else t

    def psi_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t) if self.family == 'tv' else torch.ones_like(t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.family == 'tv' else 0.5 * t.square()


def make_potential(family: str, num_basis: int = 8) -> nn.Module:
    if family == 'learned':
        return BoundedInfluencePotential(num_basis)
    return FixedPotential(family)


class InputEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MeanAggregator(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x.new_zeros(x.shape)
        src, dst = edge_index
        out = x.new_zeros(x.shape)
        out.index_add_(0, dst, x[src])
        return out / deg.unsqueeze(-1).to(x.dtype)


class HeterophilyLinearAgg(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.agg = MeanAggregator()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nb1_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.hp_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nb2_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.branch_logits = nn.Parameter(torch.zeros(4))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, edge_index_mp: torch.Tensor, deg_mp: torch.Tensor) -> torch.Tensor:
        nb1 = self.agg(h, edge_index_mp, deg_mp)
        nb2 = self.agg(nb1, edge_index_mp, deg_mp)
        hp = h - nb1

        scales = 2.0 * torch.sigmoid(self.branch_logits)
        z = (
            scales[0] * self.self_lin(h)
            + scales[1] * self.nb1_lin(nb1)
            + scales[2] * self.hp_lin(hp)
            + scales[3] * self.nb2_lin(nb2)
            + self.bias
        )
        return self.norm(z)


def symmetric_edge_features(h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    return torch.cat([h_src + h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)


def directed_edge_features(h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    return torch.cat([h_src + h_dst, h_src - h_dst, h_src * h_dst], dim=-1)


class EdgeWeightNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, init_scale: float = 0.08):
        super().__init__()
        feat_dim = 3 * in_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_scale = nn.Parameter(torch.tensor(inverse_softplus(init_scale), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, h: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return h.new_zeros((0,))

        row, col = edge_index_solver
        pair = symmetric_edge_features(h[row], h[col])
        score = self.mlp(pair).squeeze(-1)
        raw_w = F.softplus(score) + 1e-8


        raw_w = raw_w / raw_w.mean()
        scale = F.softplus(self.raw_scale)
        return scale * raw_w


class OffsetNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_relations: int = 8,
        mu_max: float = 0.75,
        init_scale: float = 0.12,
        relation_init_std: float = 0.02,
    ):
        super().__init__()
        if num_relations < 1 or mu_max <= 0 or relation_init_std <= 0:
            raise ValueError('num_relations, mu_max and relation_init_std must be positive.')
        if not 0 < init_scale < 1:
            raise ValueError('Offset scale must start in (0,1).')
        self.num_relations = num_relations
        self.mu_max = float(mu_max)
        self.relation_init_std = float(relation_init_std)

        self.relations = nn.Parameter(torch.zeros(num_relations, out_dim))
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )
        self.diff_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.raw_scale = nn.Parameter(torch.tensor(math.log(init_scale / (1.0 - init_scale)), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.relations, mean=0.0, std=self.relation_init_std)
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        nn.init.zeros_(self.diff_proj.weight)

    def directed_candidate(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(directed_edge_features(h_src, h_dst))
        pi = F.softmax(logits, dim=-1)
        return pi @ self.relations

    def forward(self, h: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return h.new_zeros((0, self.relations.size(-1)))

        row, col = edge_index_solver
        mu_tilde_ij = self.directed_candidate(h[row], h[col])
        mu_tilde_ji = self.directed_candidate(h[col], h[row])
        mu_dict = 0.5 * (mu_tilde_ij - mu_tilde_ji)
        mu_local = self.diff_proj(h[row] - h[col])
        mu = mu_dict + mu_local

        scale = torch.sigmoid(self.raw_scale)
        mu = scale * mu
        if self.mu_max > 0.0:
            mu = self.mu_max * torch.tanh(mu / self.mu_max)
        return mu


class ShiftedProxActivation(nn.Module):
    def __init__(
        self,
        potential: nn.Module,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not alpha > 0 or not 0 < kappa < 1:
            raise ValueError('Require alpha>0 and 0<kappa<1.')
        if num_pd_iter < 1 or num_newton < 1 or not 0 <= xi <= 1:
            raise ValueError('Positive solver budgets and xi in [0,1] are required.')
        self.potential = potential
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.num_pd_iter = int(num_pd_iter)
        self.num_newton = int(num_newton)
        self.xi = float(xi)
        self.eps = float(eps)

    @staticmethod
    def incidence_forward(u: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        row, col = edge_index_solver
        return u[row] - u[col]

    @staticmethod
    def incidence_adjoint(y: torch.Tensor, edge_index_solver: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index_solver
        out = y.new_zeros((num_nodes, y.size(-1)))
        out.index_add_(0, row, y)
        out.index_add_(0, col, -y)
        return out

    def step_sizes(self, max_degree: int) -> Tuple[float, float]:
        if max_degree <= 0:
            return self.kappa, self.kappa
        denom = math.sqrt(2.0 * float(max_degree))
        tau = self.kappa / denom
        sigma = self.kappa / denom
        return tau, sigma

    def edgewise_prox_unshifted(self, v: torch.Tensor, weights: torch.Tensor, sigma: float) -> torch.Tensor:
        if v.numel() == 0:
            return v
        lam = weights / sigma
        if isinstance(self.potential, FixedPotential) and self.potential.family == 'quadratic':

            return v / (1.0 + lam).unsqueeze(-1)
        r = torch.linalg.vector_norm(v, dim=-1)
        if isinstance(self.potential, FixedPotential):
            s = (r - lam).clamp_min(0.0)
        else:
            active = (r > 0.0) & (r > lam * self.potential.psi(torch.zeros_like(r)))

            r_active, lam_active = r[active], lam[active]
            s_active = r_active
            for _ in range(self.num_newton):
                numer = s_active + lam_active * self.potential.psi(s_active) - r_active
                denom = 1.0 + lam_active * self.potential.psi_prime(s_active)
                s_active = torch.minimum((s_active - numer / denom).clamp_min(0.0), r_active)
            s = torch.zeros_like(r).masked_scatter(active, s_active)

        safe_r = torch.where(r > 0.0, r, torch.ones_like(r))
        scale = torch.where(r > 0.0, s / safe_r, torch.zeros_like(r))
        return scale.unsqueeze(-1) * v

    def forward(
        self,
        z: torch.Tensor,
        edge_index_solver: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        max_degree: int,
    ) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return z

        if max_degree <= 0:
            raise ValueError('A nonempty incidence graph requires a positive maximum degree.')
        tau, sigma = self.step_sizes(max_degree)

        u = z
        u_bar = u
        y = z.new_zeros((edge_index_solver.size(1), z.size(1)))

        z_coeff = tau / self.alpha
        prox_g_denom = 1.0 + z_coeff

        for _ in range(self.num_pd_iter):
            y_tilde = y + sigma * (self.incidence_forward(u_bar, edge_index_solver) - offsets)
            y_hat = self.edgewise_prox_unshifted(y_tilde / sigma, weights, sigma)
            y_next = y_tilde - sigma * y_hat

            q = u - tau * self.incidence_adjoint(y_next, edge_index_solver, z.size(0))
            u_next = (q + z_coeff * z) / prox_g_denom
            u_bar = u_next + self.xi * (u_next - u)

            u = u_next
            y = y_next

        return u


class ShiftedProxActLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_hidden_dim: int = 128,
        num_relations: int = 8,
        num_basis: int = 8,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        mu_max: float = 0.75,
        dropout: float = 0.6,
        gamma: float = 1.0,
        relation_init_std: float = 0.02,
    ):
        super().__init__()
        self.lin_agg = HeterophilyLinearAgg(in_dim=in_dim, out_dim=out_dim)
        self.edge_weight_net = EdgeWeightNet(in_dim=in_dim, hidden_dim=edge_hidden_dim)
        self.offset_net = OffsetNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=edge_hidden_dim,
            num_relations=num_relations,
            mu_max=mu_max,
            relation_init_std=relation_init_std,
        )
        self.prox_act = ShiftedProxActivation(
            potential=BoundedInfluencePotential(num_basis=num_basis),
            alpha=alpha,
            kappa=kappa,
            num_pd_iter=num_pd_iter,
            num_newton=num_newton,
            xi=xi,
        )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must lie in [0,1].')

        self.register_buffer('prox_gate', torch.tensor(float(gamma)))
        self.residual_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    @property
    def potential(self) -> nn.Module:

        return self.prox_act.potential

    def forward(
        self,
        h: torch.Tensor,
        edge_index_mp: torch.Tensor,
        edge_index_solver: torch.Tensor,
        deg_mp: torch.Tensor,
        max_degree: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.lin_agg(h, edge_index_mp, deg_mp)
        weights = self.edge_weight_net(h, edge_index_solver)
        offsets = (self.offset_net(h, edge_index_solver) if self.offset_net is not None
                   else z.new_zeros((edge_index_solver.shape[1], z.shape[1])))
        u = self.prox_act(z, edge_index_solver, weights, offsets, max_degree)

        prox_gate = self.prox_gate

        update = u if float(prox_gate) == 1.0 else z + prox_gate * (u - z)
        update = F.dropout(update, p=self.dropout, training=self.training)
        out = self.norm(self.residual_proj(h) + update)

        aux = {
            'z': z,
            'weights': weights,
            'offsets': offsets,
            'coupling_scale': F.softplus(self.edge_weight_net.raw_scale),
            'prox_gate': prox_gate.detach(),
        }
        return out, aux


class ShiftedProxGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        edge_hidden_dim: int = 128,
        num_relations: int = 8,
        num_basis: int = 8,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        mu_max: float = 0.75,
        dropout: float = 0.6,
        gamma: float = 1.0,
        relation_init_std: float = 0.02,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.input_encoder = InputEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

        layers = []
        for _ in range(num_layers):
            layers.append(
                ShiftedProxActLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    num_relations=num_relations,
                    num_basis=num_basis,
                    alpha=alpha,
                    kappa=kappa,
                    num_pd_iter=num_pd_iter,
                    num_newton=num_newton,
                    xi=xi,
                    mu_max=mu_max,
                    dropout=dropout,
                    gamma=gamma,
                    relation_init_std=relation_init_std,
                )
            )
        self.layers = nn.ModuleList(layers)

        total_dim = hidden_dim * (num_layers + 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data) -> Dict[str, torch.Tensor]:
        h = self.input_encoder(data.x)

        reps = [h]
        aux_per_layer = []
        for layer in self.layers:
            h, aux = layer(
                h,
                data.edge_index_mp,
                data.edge_index_solver,
                data.deg_mp,
                data.max_degree,
            )
            reps.append(h)
            aux_per_layer.append(aux)

        final_rep = torch.cat(reps, dim=-1)
        logits = self.classifier(final_rep)
        return {
            'logits': logits,
            'embeddings': final_rep,
            'aux': aux_per_layer,
        }


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    edge_hidden_dim: int = 128
    num_layers: int = 2
    num_relations: int = 8
    num_basis: int = 8
    alpha: float = 1.0
    kappa: float = 0.9
    num_pd_iter: int = 12
    num_newton: int = 8
    xi: float = 1.0
    mu_max: float = 0.75
    dropout: float = 0.6
    gamma: float = 1.0

    relation_init_std: float = 0.02

    def __post_init__(self):
        for name in ('hidden_dim', 'edge_hidden_dim', 'num_layers', 'num_relations',
                     'num_basis', 'num_pd_iter', 'num_newton'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive.')
        if not (self.alpha > 0 and 0 < self.kappa < 1 and 0 <= self.xi <= 1):
            raise ValueError('Invalid solver parameters.')
        if not (0 <= self.gamma <= 1 and 0 <= self.dropout < 1):
            raise ValueError('Invalid gate or dropout.')
        if self.mu_max <= 0 or self.relation_init_std <= 0:
            raise ValueError('mu_max and relation_init_std must be positive.')


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 5e-4
    weight_reg: float = 1e-4
    offset_reg: float = 5e-5
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    max_epochs: int = 2000
    patience: int = 200
    scheduler_factor: float = 0.5
    scheduler_patience: int = 50
    min_lr: float = 1e-5
    supervised_reduction: str = 'sum'
    audit_every: int = 20
    log_every: int = 50
    deterministic: bool = False

    def __post_init__(self):
        if self.supervised_reduction not in ('sum', 'mean'):
            raise ValueError('supervised_reduction must be sum or mean.')
        if self.max_epochs < 1 or self.patience < 1 or self.scheduler_patience < 0:
            raise ValueError('Invalid epoch/patience configuration.')
        if self.lr <= 0 or self.min_lr < 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError('Invalid learning-rate configuration.')
        if min(self.weight_decay, self.weight_reg, self.offset_reg, self.grad_clip) < 0:
            raise ValueError('Decay/regularization/clipping must be nonnegative.')
        if not 0 <= self.label_smoothing < 1 or min(self.audit_every, self.log_every) < 0:
            raise ValueError('Invalid smoothing or logging interval.')


FAMILIES = ('tv', 'quadratic', 'learned')
TARGETS = ('zero', 'learned')
VARIANTS = tuple((family, target) for family in FAMILIES for target in TARGETS)


def make_base_model(in_dim: int, num_classes: int, config: ModelConfig) -> ShiftedProxGNN:
    return ShiftedProxGNN(in_dim=in_dim, num_classes=num_classes, **asdict(config))


def make_variant(base: ShiftedProxGNN, family: str = 'learned', target: str = 'learned') -> ShiftedProxGNN:
    if (family, target) not in VARIANTS:
        raise ValueError(f'Unknown variant: {(family, target)}')
    model = copy.deepcopy(base)
    for layer in model.layers:
        if family != 'learned':
            layer.prox_act.potential = FixedPotential(family)
        if target == 'zero':
            layer.offset_net = None
    model.potential_family = family
    model.target_mode = target
    return model


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    def count(params):
        return sum(p.numel() for p in params if p.requires_grad)
    return {
        'total': count(model.parameters()),
        'offset_net': sum(count(layer.offset_net.parameters()) for layer in model.layers
                          if layer.offset_net is not None),
        'weight_net': sum(count(layer.edge_weight_net.parameters()) for layer in model.layers),
        'potential': sum(count(layer.potential.parameters()) for layer in model.layers),
    }


def primary_metric_name(data_id: int) -> str:
    if data_id not in DATASET_NAMES:
        raise ValueError(f'Unknown dataset id: {data_id}')
    return 'roc_auc' if data_id == 1 else 'accuracy'


def _checked_masked_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != logits.shape[0]:
        raise ValueError('Metric mask must be a boolean vector of length n.')
    if not bool(mask.any()):
        raise ValueError('Cannot compute a metric on an empty mask.')
    if y.ndim != 1 or y.numel() != logits.shape[0]:
        raise ValueError('Labels must have shape [n].')
    selected = logits[mask]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError('Nonfinite evaluation logits.')
    return selected, y[mask]


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    selected, labels = _checked_masked_logits(logits, y, mask)
    return float(selected.argmax(dim=-1).eq(labels).double().mean().item())


@torch.no_grad()
def masked_roc_auc(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score
    selected, labels = _checked_masked_logits(logits, y, mask)
    if selected.shape[-1] != 2 or set(labels.cpu().tolist()) != {0, 1}:
        raise ValueError('Binary ROC-AUC requires logits [n,2] and both labels 0 and 1 in the mask.')

    probabilities = selected.double().softmax(dim=-1)[:, 1].cpu().numpy()
    return float(roc_auc_score(labels.cpu().numpy(), probabilities))


@torch.no_grad()
def primary_metric(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, data_id: int) -> float:

    score = masked_roc_auc(logits, y, mask) if primary_metric_name(data_id) == 'roc_auc' else masked_accuracy(logits, y, mask)
    return 100.0 * score


@torch.no_grad()
def summarize_aux(aux_per_layer):
    if not aux_per_layer:
        return 0.0, 0.0, 1.0
    last = aux_per_layer[-1]
    weights, offsets = last['weights'], last['offsets']
    return (float(weights.mean()) if weights.numel() else 0.0,
            float(offsets.norm(dim=-1).mean()) if offsets.numel() else 0.0,
            float(last['prox_gate']))


def auxiliary_regularization(aux_per_layer, weight_reg: float = 0.0, offset_reg: float = 0.0) -> torch.Tensor:
    if not aux_per_layer:
        return torch.tensor(0.0)
    reg = aux_per_layer[0]['z'].new_zeros(())
    for aux in aux_per_layer:
        if aux['weights'].numel() and weight_reg:
            reg = reg + weight_reg * aux['coupling_scale']
        if aux['offsets'].numel() and offset_reg:
            reg = reg + offset_reg * aux['offsets'].square().mean()
    return reg


def supervised_loss(logits, labels, mask, reduction='sum', label_smoothing=0.0):
    if reduction not in ('sum', 'mean'):
        raise ValueError('Only sum and mean reductions are supported.')
    selected, target = _checked_masked_logits(logits, labels, mask)
    return F.cross_entropy(selected, target, reduction=reduction, label_smoothing=label_smoothing)


def _gradient_group(parameters) -> Dict[str, Any]:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return {'norm': 0.0, 'nonfinite_elements': 0, 'tensors_with_grad': 0}
    bad = sum(int((~torch.isfinite(g)).sum()) for g in grads)
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads)) if not bad else None
    return {'norm': norm, 'nonfinite_elements': bad, 'tensors_with_grad': len(grads)}


def gradient_audit(model) -> Dict[str, Any]:
    report = {}
    for name in ('offset_net', 'edge_weight_net', 'potential'):
        modules = [getattr(layer, name) for layer in model.layers]
        report[name] = _gradient_group(p for module in modules if module is not None for p in module.parameters())
    for index, layer in enumerate(model.layers):
        if layer.offset_net is not None:
            module = layer.offset_net
            report[f'layer_{index}/relations'] = _gradient_group([module.relations])
            report[f'layer_{index}/mixture_last'] = _gradient_group(module.mlp[-1].parameters())
            report[f'layer_{index}/local_projection'] = _gradient_group(module.diff_proj.parameters())
    return report


def train_one_epoch(model, data, optimizer, grad_clip=1.0, weight_reg=1e-4,
                    offset_reg=5e-5, label_smoothing=0.0,
                    supervised_reduction='sum', collect_audit=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(data)
    logits = out['logits']
    sup = supervised_loss(logits, data.y, data.train_mask, supervised_reduction, label_smoothing)
    reg = auxiliary_regularization(out['aux'], weight_reg, offset_reg)
    loss = sup + reg
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training loss; refusing to continue.')
    loss.backward()
    audit = gradient_audit(model) if collect_audit else None
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip if grad_clip > 0 else float('inf'), error_if_nonfinite=True)
    optimizer.step()
    mean_w, mean_mu, gate = summarize_aux(out['aux'])

    return {'loss': float(loss.detach()), 'supervised_loss': float(sup.detach()),
            'regularization': float(reg.detach()), 'pre_clip_norm': float(pre_clip_norm),
            'clipped': bool(grad_clip > 0 and pre_clip_norm > grad_clip),
            'mean_weight_last': mean_w, 'mean_target_norm_last': mean_mu,
            'gate_last': gate, 'gradient_audit': audit}


@torch.no_grad()
def evaluate(model, data, data_id: int, include_test: bool = False):

    model.eval()
    out = model(data)
    result = {'metric': primary_metric_name(data_id),
              'val_score': primary_metric(out['logits'], data.y, data.val_mask, data_id)}
    if include_test:
        result['test_score'] = primary_metric(out['logits'], data.y, data.test_mask, data_id)
    return result


def fit(model, data, data_id: int, seed: int, config: TrainConfig,
        on_epoch: Optional[Callable[[Dict[str, Any]], None]] = None):

    set_seed(seed, config.deterministic)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.min_lr)
    best_val, best_epoch, best_state, bad_epochs = -float('inf'), 0, None, 0
    sampled_audits = []
    for epoch in range(1, config.max_epochs + 1):
        collect = config.audit_every > 0 and epoch % config.audit_every == 0
        info = train_one_epoch(model, data, optimizer, config.grad_clip, config.weight_reg,
                               config.offset_reg, config.label_smoothing,
                               config.supervised_reduction, collect)
        val = evaluate(model, data, data_id, include_test=False)['val_score']
        if not math.isfinite(val):
            raise FloatingPointError(f'Nonfinite validation metric at epoch {epoch}.')
        scheduler.step(val)
        improved = val > best_val
        if improved:
            best_val, best_epoch = val, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        info.update(epoch=epoch, val_score=val, best_val_score=best_val,
                    best_epoch=best_epoch, improved=improved, lr=float(optimizer.param_groups[0]['lr']))
        if collect:
            sampled_audits.append({'epoch': epoch, 'pre_clip_norm': info['pre_clip_norm'],
                                   'clipped': info['clipped'], 'modules': info['gradient_audit']})
        if on_epoch:
            on_epoch(info)

        if bad_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError('No validation-selected state was obtained.')
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, data, data_id, include_test=True)
    final.update(best_epoch=best_epoch, epochs_trained=epoch,
                 parameter_counts=parameter_counts(model), gradient_samples=sampled_audits,
                 sampled_clipping_rate=(sum(a['clipped'] for a in sampled_audits) / len(sampled_audits)
                                        if sampled_audits else None))
    return final, best_state


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def dataset_record(data, data_id: int, split_idx: int) -> Dict[str, Any]:
    return {'dataset_id': data_id, 'dataset': DATASET_NAMES[data_id], 'split_idx': split_idx,
            'num_nodes': int(data.num_nodes), 'num_features': int(data.x.shape[1]),
            'solver_edges': int(data.edge_index_solver.shape[1]),
            'mp_edges': int(data.edge_index_mp.shape[1]), 'max_degree': int(data.max_degree),
            'hashes': {name: tensor_fingerprint(getattr(data, name)) for name in
                       ('x', 'y', 'edge_index_solver', 'train_mask', 'val_mask', 'test_mask')}}


def environment_record() -> Dict[str, Any]:
    versions = {}
    for name in ('torch', 'numpy', 'scikit-learn', 'torch-geometric'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'versions': versions, 'cuda_available': torch.cuda.is_available(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_prepared_data(data_id: int, split_idx: int, data_root='/tmp'):
    dataset, num_classes = load_dataset(data_id, data_root)
    data = dataset[0]


    expected = {0: (22662, 300, 18), 1: (10000, 7, 2), 2: (24492, 300, 5),
                3: (2277, 2325, 5), 4: (5201, 2089, 5)}
    got = (int(data.num_nodes), int(data.x.shape[1]), int(num_classes))
    if data_id in expected and got != expected[data_id]:
        raise ValueError(f'Dataset instance mismatch: got {got}, manuscript lists {expected[data_id]}. '
                         'Use the intended dataset files; no automatic variant substitution.')
    data = pick_single_split(data, split_idx)
    data = prepare_graph(data)
    data.x = F.normalize(data.x.to(torch.float32), p=2, dim=-1)
    data.y = data.y.reshape(-1).to(torch.long)
    return data, num_classes


def write_json(path: Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def add_config_arguments(parser):
    for name, value in asdict(ModelConfig()).items():
        parser.add_argument('--' + name, type=type(value), default=value)
    for name, value in asdict(TrainConfig()).items():
        if name == 'deterministic':
            parser.add_argument('--deterministic', action='store_true',
                                help='Require deterministic PyTorch kernels; unsupported kernels raise.')
        elif name == 'supervised_reduction':
            parser.add_argument('--supervised_reduction', choices=('sum', 'mean'), default=value,
                                help='sum follows the manuscript equation; mean is explicitly off-equation.')
        else:
            parser.add_argument('--' + name, type=type(value), default=value)


def configs_from_args(args):
    mc = ModelConfig(**{name: getattr(args, name) for name in asdict(ModelConfig())})
    tc = TrainConfig(**{name: getattr(args, name) for name in asdict(TrainConfig())})
    return mc, tc


def main(argv=None):
    parser = argparse.ArgumentParser(description='Manuscript-aligned SCALET single-run training')
    parser.add_argument('data', type=int, choices=range(9), help='dataset id 0..8, as in the original script')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data_root', default='/tmp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--potential', choices=FAMILIES, default='learned')
    parser.add_argument('--target', choices=TARGETS, default='learned')
    parser.add_argument('--out_dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true', help='Explicitly allow overwriting this run directory.')
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    mc, tc = configs_from_args(args)
    out_dir = args.out_dir or Path('runs') / DATASET_NAMES[args.data] / f'{args.potential}_{args.target}_split{args.split_idx}_seed{args.seed}'
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(f'{out_dir} is not empty. Use a new directory or --force.')
    out_dir.mkdir(parents=True, exist_ok=True)
    if tc.supervised_reduction != 'sum':
        print('[Non-manuscript setting] supervised_reduction=mean; the manuscript equation uses sum.')
    data, num_classes = load_prepared_data(args.data, args.split_idx, args.data_root)
    record = dataset_record(data, args.data, args.split_idx)
    device = get_device(args.device)
    set_seed(args.seed, tc.deterministic)
    base = make_base_model(data.x.shape[1], num_classes, mc)
    model = make_variant(base, args.potential, args.target).to(device)
    del base
    data = data.to(device)
    metadata = {'schema_version': 1, 'implementation': 'manuscript-aligned-repair',
                'dataset': record, 'model_config': asdict(mc), 'train_config': asdict(tc),
                'seed': args.seed, 'family': args.potential, 'target': args.target,
                'device': str(device), 'environment': environment_record(),
                'initialization_choice': 'R~Normal(0,relation_init_std^2), mixture last layer=0, local D=0',
                'paper_scores_reproduced': False}
    write_json(out_dir / 'run_config.json', metadata)
    print(f'{DATASET_NAMES[args.data]} | {args.potential}/{args.target} | '
          f'{primary_metric_name(args.data)} (0--100) | device={device}')
    print(f'Trainable parameters: {parameter_counts(model)}')
    with (out_dir / 'training.jsonl').open('w', encoding='utf-8') as stream:
        def on_epoch(info):
            stream.write(json.dumps(info, allow_nan=False) + '\n')
            stream.flush()
            if tc.log_every and (info['epoch'] % tc.log_every == 0 or info['improved']):
                print(f"Epoch {info['epoch']:04d} | Loss {info['loss']:.6f} | "
                      f"Val {info['val_score']:.6f} | Best {info['best_val_score']:.6f} | "
                      f"Grad {info['pre_clip_norm']:.4g}")
        result, state = fit(model, data, args.data, args.seed, tc, on_epoch)
    torch.save({'state_dict': state, 'metadata': metadata, 'best_epoch': result['best_epoch']},
               out_dir / 'best_checkpoint.pt')
    write_json(out_dir / 'result.json', {**metadata, 'result': result})
    print(f"Selected epoch {result['best_epoch']} | Val {result['val_score']:.6f} | "
          f"Test {result['test_score']:.6f} ({result['metric']})")


if __name__ == '__main__':
    main()
