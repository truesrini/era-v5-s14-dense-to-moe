"""
What did the experts learn? (Session 14, s.9)  Route validation text through the trained MoE and report
  * for a few layers, the tokens each expert receives most consistently
  * how often consecutive tokens share their top-1 expert, against the 1/E chance rate

    python analyze_experts.py --ckpt checkpoints/moe.pt
"""
import argparse
from collections import Counter

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/moe.pt")
p.add_argument("--batches", type=int, default=60)
p.add_argument("--min_count", type=int, default=200)
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
cfg = ModelConfig(**ck["config"])
model = GPT(cfg).cuda()
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("data/tokenizer.json")
val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
T, E, V = cfg.block_size, cfg.n_experts, cfg.vocab_size
L = cfg.n_layer

counts = torch.zeros(L, V, E, device="cuda")          # times token v was routed to expert e in layer l
tok_total = torch.zeros(V, device="cuda")
same_top1 = torch.zeros(L, device="cuda")
pairs = 0
rng = np.random.default_rng(7)
with torch.no_grad():
    for _ in range(args.batches):
        offs = rng.integers(0, len(val) - T - 1, 32)
        x = torch.from_numpy(np.stack([val[i:i + T] for i in offs]).astype(np.int64)).cuda()
        model(x)
        ids = x.reshape(-1)
        tok_total += torch.bincount(ids, minlength=V).float()
        for li, m in enumerate(model.moe_layers()):
            topk = m.last_topk                                         # (N, k), sorted by choice score
            for j in range(cfg.top_k):
                counts[li].index_put_((ids, topk[:, j]), torch.ones_like(ids, dtype=torch.float), accumulate=True)
            t1 = topk[:, 0].view(32, T)
            same_top1[li] += (t1[:, 1:] == t1[:, :-1]).float().sum()
        pairs += 32 * (T - 1)

lines = ["# What the experts learned", "",
         f"Routed {args.batches * 32 * T:,} validation tokens through `{args.ckpt}`.", "",
         "## Consecutive tokens sharing their top-1 expert", "",
         f"Chance level with {E} experts is 1/{E} = {100 / E:.2f}%.", "",
         "| layer | consecutive tokens with the same top-1 expert |", "|---|---|"]
for li in range(L):
    lines.append(f"| {li} | {100 * same_top1[li].item() / pairs:.1f}% |")

frequent = (tok_total >= args.min_count).nonzero().squeeze(1)


def show(v):
    s = tok.decode([int(v)])
    if not s.strip() or "�" in s:            # partial UTF-8 byte or bare whitespace: show the BPE symbol
        s = tok.id_to_token(int(v))
    return "`" + s.replace("\n", "\\n").replace("|", "\\|").replace("`", "'") + "`"


for li in sorted({0, L // 2, L - 1}):
    lines += ["", f"## Layer {li}: tokens each expert receives most consistently", "",
              f"For every token seen at least {args.min_count} times, the share of its occurrences routed to the expert "
              f"(each token goes to {cfg.top_k} experts, so an even spread would be {100 * cfg.top_k / E:.1f}%).", "",
              "| expert | load | top tokens (share of that token's occurrences) |", "|---|---|---|"]
    share = counts[li][frequent] / tok_total[frequent][:, None]       # (n_frequent, E)
    load = counts[li].sum(0) / counts[li].sum()
    for e in range(E):
        top = share[:, e].topk(8)
        items = [f"{show(frequent[i])} {100 * s:.0f}%" for s, i in zip(top.values.tolist(), top.indices.tolist())]
        lines.append(f"| {e} | {100 * load[e].item():.1f}% | " + ", ".join(items) + " |")

out = "\n".join(lines) + "\n"
open("logs/expert_specialization.md", "w", encoding="utf-8").write(out)
print(out)
