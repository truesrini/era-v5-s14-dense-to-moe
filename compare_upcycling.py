"""
How much does each way of turning the dense FFN into experts disturb the model at the moment of
conversion? No training here: build each MoE from the same dense checkpoint and measure val loss.

    python compare_upcycling.py --init checkpoints/dense.pt
"""
import argparse
import json

import numpy as np
import torch

import upcycle as up
from model import GPT, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--init", default="checkpoints/dense.pt")
p.add_argument("--eval_batches", type=int, default=40)
args = p.parse_args()

device = "cuda"
ck = torch.load(args.init, map_location=device, weights_only=False)
dcfg = ModelConfig(**ck["config"])
dense = GPT(dcfg).to(device)
dense.load_state_dict(ck["model"])
T = dcfg.block_size
val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
train = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
offs = np.random.default_rng(0).integers(0, len(val) - T - 1, (args.eval_batches, 32))


def batch(data, o):
    x = torch.from_numpy(np.stack([data[i:i + T + 1] for i in o]).astype(np.int64)).to(device)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(model):
    model.eval()
    out = []
    for o in offs:
        x, y = batch(val, o)
        with torch.autocast("cuda", enabled=False):
            out.append(model(x, y)[1].item())
    return float(np.mean(out))


crng = np.random.default_rng(123)
calib = [batch(train, crng.integers(0, len(train) - T - 1, 32))[0] for _ in range(8)]
importance = up.neuron_importance(dense, calib)
H = dcfg.ffn_hidden
base = {**dcfg.to_dict(), "moe": True, "n_experts": 16, "top_k": 2, "expert_hidden": 256,
        "shared_hidden": 512, "route_scale": 2.0}
variants = [
    ("copy: 8 full copies, top-2 (sparse upcycling)", "copy",
     {**base, "n_experts": 8, "expert_hidden": H, "shared_hidden": 0, "route_scale": 1.0}, None, 0.0),
    ("partition, random neurons -> shared, random subsets -> experts", "partition", base, None, 0.0),
    ("partition, important neurons -> shared (ours)", "partition", base, importance, 0.0),
    ("partition (ours) + drop-upcycling r=0.5", "partition", base, importance, 0.5),
    ("partition (ours), route_scale 1 instead of 2", "partition", {**base, "route_scale": 1.0}, importance, 0.0),
    ("no shared expert: 16 experts x 512, top-2", "partition",
     {**base, "shared_hidden": 0, "expert_hidden": 512, "route_scale": 2.0}, importance, 0.0),
    ("scratch: attention copied, FFN re-initialised", "scratch", base, None, 0.0),
]
dense_loss = evaluate(dense)
rows = [{"method": "dense model (before conversion)", "val_loss": dense_loss, "delta": 0.0,
         "total_params_M": dense.param_counts()[0] / 1e6, "active_params_M": dense.param_counts()[1] / 1e6}]
for name, method, cd, imp, drop in variants:
    cfg = ModelConfig(**cd)
    plan = up.make_plan(dcfg, cfg, method, imp, drop, seed=0)
    moe = up.upcycle_model(dense, cfg, plan)
    v = evaluate(moe)
    tot, act = moe.param_counts()
    rows.append({"method": name, "val_loss": v, "delta": v - dense_loss,
                 "total_params_M": tot / 1e6, "active_params_M": act / 1e6})
    del moe
    torch.cuda.empty_cache()

lines = ["| method | val loss | change | total params | active params |", "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['method']} | {r['val_loss']:.4f} | {r['delta']:+.4f} | "
                 f"{r['total_params_M']:.1f}M | {r['active_params_M']:.1f}M |")
table = "\n".join(lines)
print(table)
open("logs/upcycle_conversion.md", "w", encoding="utf-8").write(table + "\n")
json.dump(rows, open("logs/upcycle_conversion.json", "w"), indent=2)
