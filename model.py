"""
A small Qwen3-style decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU) whose
feed-forward block is either one dense SwiGLU network or a mixture-of-experts layer.

The MoE layer follows the recipe from Session 14:
  * router = one matrix d_model x n_experts, run in fp32, started at a small scale
  * sigmoid scores, top-k, the kept scores rescaled to sum to 1, times a routed scaling factor
  * one shared expert that every token uses, plus routed experts
  * dropless: every token is processed by every expert it picks (no capacity, no dropping)
  * auxiliary-loss-free balancing: a per-expert bias added to the scores only when choosing,
    moved by +-gamma after every optimizer step, with load counted over the whole batch
  * a very small sequence-level auxiliary loss (alpha ~ 1e-4) against extreme imbalance
  * optional probabilistic (Gumbel top-k) selection for an early window after upcycling,
    so near-identical clone experts all receive gradient before the router settles
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_kv_head: int = 2
    d_model: int = 256
    ffn_hidden: int = 1024          # dense SwiGLU inner width
    rope_theta: float = 10000.0
    # --- mixture of experts (used when moe=True) ---
    moe: bool = False
    n_experts: int = 16             # routed experts per layer
    top_k: int = 2
    expert_hidden: int = 256        # inner width of one routed expert
    shared_hidden: int = 512        # inner width of the shared expert (0 = none)
    score_func: str = "sigmoid"     # "sigmoid" or "softmax"
    route_scale: float = 2.0        # routed scaling factor
    bias_update_speed: float = 1e-3 # gamma of aux-loss-free balancing (0 = off)
    seq_aux_alpha: float = 1e-4     # sequence-level balance loss weight (0 = off)
    router_init_std: float = 0.002  # ~1/10 of the usual 0.02

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cache(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                        # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd), rotate pairs (first half, second half)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd)   # Qwen3 QK-norm
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # GQA: each group of n_head / n_kv query heads reads one shared K/V head
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        E, d, h = cfg.n_experts, cfg.d_model, cfg.expert_hidden
        self.router = nn.Linear(d, E, bias=False)
        # routed experts stored as stacked weights: (E, out, in), same layout as nn.Linear
        self.w_gate = nn.Parameter(torch.empty(E, h, d))
        self.w_up = nn.Parameter(torch.empty(E, h, d))
        self.w_down = nn.Parameter(torch.empty(E, d, h))
        self.shared = SwiGLU(d, cfg.shared_hidden) if cfg.shared_hidden > 0 else None
        # aux-loss-free balancing state (buffers, never touched by the optimizer)
        self.register_buffer("expert_bias", torch.zeros(E))
        self.register_buffer("load_accum", torch.zeros(E))   # tokens per expert, whole batch
        self.sample_noise = 0.0   # > 0: probabilistic selection (Gumbel noise scale)
        self.last_aux = None
        self.last_topk = None     # (B*T, k) expert choices of the last forward, for analysis

    @torch.no_grad()
    def update_bias(self):
        """Called once per optimizer step: b_i += gamma * sign(mean_load - load_i)."""
        load = self.load_accum
        if self.cfg.bias_update_speed > 0 and load.sum() > 0:
            self.expert_bias += self.cfg.bias_update_speed * torch.sign(load.mean() - load)
        out = load.clone()
        self.load_accum.zero_()
        return out

    def forward(self, x):
        cfg = self.cfg
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N, E, k = xf.shape[0], cfg.n_experts, cfg.top_k

        # ---- router in fp32 ----
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(xf.float(), self.router.weight.float())     # (N, E)
            if cfg.score_func == "sigmoid":
                scores = torch.sigmoid(logits)
            else:
                scores = torch.softmax(logits, dim=-1)
            choose = scores + self.expert_bias                             # bias only for choosing
            if self.training and self.sample_noise > 0:
                g = -torch.log(-torch.log(torch.rand_like(choose).clamp_(1e-9, 1 - 1e-9)))
                choose = choose + self.sample_noise * g                    # Gumbel top-k sampling
            topk_idx = choose.topk(k, dim=-1).indices                       # (N, k)
            self.last_topk = topk_idx.detach()
            topk_w = scores.gather(1, topk_idx)                             # weights from raw scores
            topk_w = topk_w / topk_w.sum(-1, keepdim=True) * cfg.route_scale

            # sequence-level balance loss (DeepSeek-V3 style), tiny weight
            aux = None
            if self.training and cfg.seq_aux_alpha > 0:
                s = scores.view(B, T, E)
                s = s / s.sum(-1, keepdim=True)
                P = s.mean(1)                                               # (B, E)
                onehot = torch.zeros(B, T * k, E, device=x.device)
                onehot.scatter_(2, topk_idx.view(B, T * k, 1), 1.0)
                f = onehot.sum(1) * E / (k * T)                             # (B, E)
                aux = cfg.seq_aux_alpha * (f * P).sum(-1).mean()
            self.last_aux = aux

        # ---- dropless dispatch: sort token copies by expert, run each expert on its slice ----
        flat = topk_idx.reshape(-1)                                         # (N*k,)
        order = flat.argsort(stable=True)
        tok = order // k
        counts = torch.bincount(flat, minlength=E)
        if self.training:
            self.load_accum += counts.float()
        xs = xf[tok]
        ws = topk_w.reshape(-1)[order].to(xf.dtype)
        out = torch.empty_like(xs)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            xe = xs[start:start + n]
            he = F.silu(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            out[start:start + n] = he @ self.w_down[e].t()
            start += n
        y = torch.zeros_like(xf).index_add_(0, tok, out * ws[:, None])
        if self.shared is not None:
            y = y + self.shared(xf)
        return y.view(B, T, D)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = MoE(cfg) if cfg.moe else SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # tied embeddings
        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():               # scaled init of residual projections
            if n.endswith("o.weight") or n.endswith("w_down.weight") or n.endswith(".w_down"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, MoE):
            nn.init.normal_(m.w_gate, 0.0, 0.02)
            nn.init.normal_(m.w_up, 0.0, 0.02)
            nn.init.normal_(m.router.weight, 0.0, m.cfg.router_init_std)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        aux = None
        for m in self.moe_layers():
            if m.last_aux is not None:
                aux = m.last_aux if aux is None else aux + m.last_aux
        return logits, loss, aux

    def param_counts(self):
        """Total parameters and parameters active for one token (embeddings counted once)."""
        total = sum(p.numel() for p in self.parameters())
        if not self.cfg.moe:
            return total, total
        c = self.cfg
        per_expert = 3 * c.d_model * c.expert_hidden
        inactive = len(self.moe_layers()) * (c.n_experts - c.top_k) * per_expert
        return total, total - inactive

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
