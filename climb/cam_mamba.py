from __future__ import annotations

import math
from inspect import isfunction
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def exists(val) -> bool:
    return val is not None


def is_empty(t: torch.Tensor) -> bool:
    return t.nelement() == 0


def expand_dim(t: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    return d() if (not exists(x) and isfunction(d)) else (d if not exists(x) else x)


def ema(old: Optional[torch.Tensor], new: torch.Tensor, decay: float) -> torch.Tensor:
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bld,cd->blc", x, means)


def dists_and_buckets(x: torch.Tensor, means: torch.Tensor):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index: torch.Tensor, num_classes: int, dim: int = -1) -> torch.Tensor:
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x: torch.Tensor, means: torch.Tensor, buckets: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size, _, dim = x.shape
    dtype = x.dtype
    cluster_num = means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, cluster_num).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(batch_size, cluster_num, dim, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, dim), x)

    means_ = means_.sum(0) / bins.clamp(min=1).squeeze(0).unsqueeze(-1)
    means_ = F.normalize(means_, dim=-1).type(dtype)

    means = torch.where(zero_mask.squeeze(0).unsqueeze(-1), means, means_)
    return means


def index_reverse(index: torch.Tensor) -> torch.Tensor:
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def apply_permute(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    dim = index.dim()
    assert x.shape[:dim] == index.shape, f"x ({x.shape}) and index ({index.shape}) shape incompatible"

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    return torch.gather(x, dim=dim - 1, index=index)


class TokenPermutation:
    def __init__(self, stable: bool = True):
        self.stable = stable

    def sort_indices(self, cluster_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, sort_idx = torch.sort(cluster_index, dim=-1, stable=self.stable)
        inv_idx = index_reverse(sort_idx)
        return sort_idx, inv_idx


class PromptingOps:
    def __init__(self, cluster_num: int, ema_decay: float, n_iter: int):
        self.cluster_num = cluster_num
        self.ema_decay = ema_decay
        self.n_iter = n_iter

    def update_centroids(
        self,
        x: torch.Tensor,
        means: torch.Tensor,
        initted: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, num_tokens, _ = x.shape
        cluster_num = self.cluster_num

        if not bool(initted.item()):
            pad_n = (cluster_num - num_tokens % cluster_num) % cluster_num
            padded_x = F.pad(x, (0, 0, 0, pad_n))
            x_means = torch.mean(
                rearrange(padded_x, "b (cnt n) c -> cnt (b n) c", cnt=cluster_num),
                dim=-2,
            ).detach()
        else:
            x_means = means.detach()

        if training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        x_means = x_means.detach()

        if training:
            with torch.no_grad():
                if not bool(initted.item()):
                    means.data.copy_(x_means)
                    initted.data.copy_(torch.tensor(True, device=initted.device))
                else:
                    ema_inplace(means, x_means, self.ema_decay)

        return x_means, initted

    @staticmethod
    def hard_assign(x: torch.Tensor, x_means: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores = torch.einsum("b i c, j c -> b i j", F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
            return torch.argmax(scores, dim=-1)


class Selective_Scan(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 2.0,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        num_scans = 1

        xs = x.permute(0, 2, 1).view(batch_size, 1, -1, seq_len).contiguous()
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch_size, num_scans, -1, seq_len), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, num_scans, -1, seq_len), self.dt_projs_weight)

        xs = xs.float().view(batch_size, -1, seq_len)
        dts = dts.contiguous().float().view(batch_size, -1, seq_len)
        Bs = Bs.float().view(batch_size, num_scans, -1, seq_len)
        Cs = Cs.float().view(batch_size, num_scans, -1, seq_len) + prompt

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch_size, num_scans, -1, seq_len)

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len, channels = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(batch_size, 1, channels, seq_len)
        y = self.forward_core(x, prompt)
        return y.permute(0, 2, 1).contiguous()


class CAM(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        cluster_num: int = 64,
        inner_rank: int = 128,
        mlp_ratio: float = 2.0,
        n_iter: int = 5,
        ema_decay: float = 0.999,
        permute_stable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.cluster_num = cluster_num
        self.inner_rank = inner_rank
        self.d_state = d_state

        self.permutation = TokenPermutation(stable=permute_stable)
        self.prompting = PromptingOps(cluster_num=cluster_num, ema_decay=ema_decay, n_iter=n_iter)

        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))
        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))

        self.register_buffer("means", torch.randn(cluster_num, dim))
        self.register_buffer("initted", torch.tensor(False))

        self.cal_embedding = nn.Linear(dim, d_state)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        return_cluster: bool = False,
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        height, width = x_size
        if num_tokens != height * width:
            raise ValueError("CAM expected {} patch tokens, got {}".format(height * width, num_tokens))

        x2d = x.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()
        x_tokens = x2d.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()

        x_means, _ = self.prompting.update_centroids(
            x=x_tokens,
            means=self.means,
            initted=self.initted,
            training=self.training,
        )
        cluster_idx = self.prompting.hard_assign(x_tokens, x_means)
        cls_policy = F.one_hot(cluster_idx, num_classes=self.cluster_num).float()

        full_embedding = self.cal_embedding(x_means)
        prompt = torch.matmul(cls_policy, full_embedding)

        sort_idx, inv_idx = self.permutation.sort_indices(cluster_idx)

        x_m = x_tokens.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()
        x_m = self.in_proj(x_m)
        x_m = x_m * torch.sigmoid(self.CPE(x_m))
        hidden = x_m.shape[1]
        x_m = x_m.view(batch_size, hidden, -1).permute(0, 2, 1).contiguous()

        semantic_x = apply_permute(x_m, sort_idx)
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x_out = apply_permute(y, inv_idx)
        if return_cluster:
            return x_out, cluster_idx
        return x_out
