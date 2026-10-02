"""
Grow a trained dense model into a mixture-of-experts model ("upcycling", Session 14 s.15).

Dense SwiGLU FFN with H inner neurons:   y = sum_j  w_down[:, j] * silu(w_gate[j] x) * (w_up[j] x)
Every inner neuron j is an independent (gate row, up row, down column) triple, so an expert can be
built from any subset of neurons. Methods:

  copy       every routed expert is a full copy of the dense FFN, no shared expert, route_scale 1.
             Exactly function-preserving (weights sum to 1), but each token now runs top_k full FFNs.
  partition  (default, Lightning-LM / Qwen1.5-MoE style)
             - rank neurons by importance  E|h_j| * ||w_down[:, j]||  on a few calibration batches
             - the top `shared_hidden` neurons become the always-on shared expert
             - the remaining C neurons are the pool for routed experts; each expert takes a random
               `expert_hidden`-subset of the pool (overlapping between experts, so they start different)
             - route_scale = |C| / expert_hidden keeps the expected layer output equal to the dense
               output when the router is still uniform, so the loss barely moves at conversion
             - optional drop-upcycling: redraw a fraction r of each expert's neurons from scratch
  scratch    attention/embeddings copied, FFN experts freshly initialised (control).

The same neuron plan is applied to the AdamW moments, so copied neurons keep their optimizer state.
"""
import math

import torch

from model import GPT, MoE, SwiGLU


@torch.no_grad()
def neuron_importance(model, batches):
    """Per dense layer: mean |h_j| over calibration tokens times the norm of w_down[:, j]."""
    sums, hooks = {}, []
    for i, b in enumerate(model.blocks):
        assert isinstance(b.ffn, SwiGLU)
        sums[i] = torch.zeros(b.ffn.w_down.in_features, device=b.ffn.w_down.weight.device)

        def hook(mod, inp, out, i=i):
            sums[i] += inp[0].float().abs().reshape(-1, inp[0].shape[-1]).mean(0)
        hooks.append(b.ffn.w_down.register_forward_hook(hook))
    model.eval()
    for x in batches:
        model(x)
    for h in hooks:
        h.remove()
    return [sums[i] / len(batches) * model.blocks[i].ffn.w_down.weight.float().norm(dim=0)
            for i in range(len(model.blocks))]


def make_plan(dense_cfg, moe_cfg, method, importance=None, drop_ratio=0.0, seed=0):
    """Which dense neurons go into the shared expert and into each routed expert, per layer."""
    g = torch.Generator().manual_seed(seed)
    H, E, He, Hs = dense_cfg.ffn_hidden, moe_cfg.n_experts, moe_cfg.expert_hidden, moe_cfg.shared_hidden
    plan = []
    for layer in range(dense_cfg.n_layer):
        if method == "scratch":
            plan.append(None)
            continue
        if method == "copy":
            assert He == H and Hs == 0, "copy needs expert_hidden == ffn_hidden and no shared expert"
            shared, pool = None, torch.arange(H)
            experts = [torch.arange(H) for _ in range(E)]
        elif method == "partition":
            if importance is not None:
                order = importance[layer].cpu().argsort(descending=True)
            else:
                order = torch.randperm(H, generator=g)
            shared, pool = (order[:Hs] if Hs > 0 else None), order[Hs:]
            assert He <= len(pool)
            experts = [pool[torch.randperm(len(pool), generator=g)[:He]] for _ in range(E)]
        else:
            raise ValueError(method)
        n_drop = int(round(drop_ratio * He))
        drops = []
        for _ in range(E):
            m = torch.zeros(He, dtype=torch.bool)
            m[torch.randperm(He, generator=g)[:n_drop]] = True
            drops.append(m)
        plan.append({"shared": shared, "experts": experts, "drop": drops, "pool": len(pool)})
    return plan


def _apply_layer(wg, wu, wd, p, fresh_in, fresh_out):
    """Slice one dense FFN's (H,d),(H,d),(d,H) tensors into shared + stacked expert tensors."""
    out = {}
    if p["shared"] is not None:
        s = p["shared"].to(wg.device)
        out["shared.w_gate.weight"] = wg[s].clone()
        out["shared.w_up.weight"] = wu[s].clone()
        out["shared.w_down.weight"] = wd[:, s].clone()
    g, u, d = [], [], []
    for idx, drop in zip(p["experts"], p["drop"]):
        idx, drop = idx.to(wg.device), drop.to(wg.device)
        eg, eu, ed = wg[idx].clone(), wu[idx].clone(), wd[:, idx].clone()
        if drop.any():
            eg[drop] = fresh_in((int(drop.sum()), wg.shape[1]))
            eu[drop] = fresh_in((int(drop.sum()), wg.shape[1]))
            ed[:, drop] = fresh_out((wd.shape[0], int(drop.sum())))
        g.append(eg), u.append(eu), d.append(ed)
    out["w_gate"], out["w_up"], out["w_down"] = torch.stack(g), torch.stack(u), torch.stack(d)
    return out


@torch.no_grad()
def upcycle_model(dense, moe_cfg, plan):
    device = next(dense.parameters()).device
    moe = GPT(moe_cfg).to(device)
    dsd, msd = dense.state_dict(), moe.state_dict()
    for k, v in dsd.items():                       # attention, norms, embeddings: copied as is
        if ".ffn." not in k:
            msd[k].copy_(v)
    std_out = 0.02 / math.sqrt(2 * moe_cfg.n_layer)
    for i, p in enumerate(plan):
        if p is None:
            continue
        pre = f"blocks.{i}.ffn."
        sl = _apply_layer(dsd[pre + "w_gate.weight"], dsd[pre + "w_up.weight"], dsd[pre + "w_down.weight"], p,
                          lambda s: torch.randn(s, device=device) * 0.02,
                          lambda s: torch.randn(s, device=device) * std_out)
        for k, v in sl.items():
            msd[pre + k].copy_(v)
    moe.load_state_dict(msd)
    return moe


@torch.no_grad()
def transfer_adam_state(dense, dense_opt, moe, moe_opt, plan):
    """Carry AdamW moments over: same tensors for copied parts, neuron-sliced for the FFN."""
    dparams, mparams = dict(dense.named_parameters()), dict(moe.named_parameters())
    n = 0
    for name, mp in mparams.items():
        if ".ffn." not in name and name in dparams:
            st = dense_opt.state.get(dparams[name])
            if st:
                moe_opt.state[mp] = {k: v.clone() for k, v in st.items()}
                n += 1
    for i, p in enumerate(plan):
        if p is None:
            continue
        pre = f"blocks.{i}.ffn."
        sts = [dense_opt.state.get(dparams[pre + w + ".weight"]) for w in ("w_gate", "w_up", "w_down")]
        if not all(sts):
            continue
        step = sts[0]["step"].clone()
        zeros = lambda s: torch.zeros(s, device=sts[0]["exp_avg"].device)
        sliced = {key: _apply_layer(sts[0][key], sts[1][key], sts[2][key], p, zeros, zeros)
                  for key in ("exp_avg", "exp_avg_sq")}
        for k in sliced["exp_avg"]:
            mp = mparams[pre + k]
            moe_opt.state[mp] = {"step": step.clone(), "exp_avg": sliced["exp_avg"][k],
                                 "exp_avg_sq": sliced["exp_avg_sq"][k]}
            n += 1
    return n
