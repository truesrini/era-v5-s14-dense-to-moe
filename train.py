"""
Train a dense model, then grow it into a mixture-of-experts model and keep training.

  phase 1   python train.py --mode dense          --run dense      --steps 3000
  phase 2a  python train.py --mode dense_continue --run dense_cont --init checkpoints/dense.pt --steps 3000
  phase 2b  python train.py --mode moe            --run moe        --init checkpoints/dense.pt --steps 3000

Phase 2a (keep training the dense model) is the baseline that phase 2b has to beat: same data
stream, same learning-rate schedule, same compute per token.

Every run writes logs/<run>.log (human readable) and logs/<run>.jsonl (one JSON record per line).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import upcycle as up
from model import GPT, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dense", "dense_continue", "moe"], required=True)
p.add_argument("--run", required=True)
p.add_argument("--init", default=None, help="dense checkpoint to continue from / upcycle")
p.add_argument("--steps", type=int, required=True)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min_lr_frac", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--decay_frac", type=float, default=0.0, help="WSD: last fraction of steps decays the lr")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_batches", type=int, default=40)
p.add_argument("--log_interval", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data_seed", type=int, default=1)
# dense shape (mode dense)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--d_model", type=int, default=256)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_kv_head", type=int, default=2)
p.add_argument("--ffn_hidden", type=int, default=1024)
p.add_argument("--block_size", type=int, default=256)
# MoE (mode moe)
p.add_argument("--upcycle", choices=["partition", "copy", "scratch"], default="partition")
p.add_argument("--importance", type=int, default=1, help="rank neurons by importance (else random)")
p.add_argument("--drop_ratio", type=float, default=0.0)
p.add_argument("--n_experts", type=int, default=16)
p.add_argument("--top_k", type=int, default=2)
p.add_argument("--expert_hidden", type=int, default=256)
p.add_argument("--shared_hidden", type=int, default=512)
p.add_argument("--route_scale", type=float, default=None, help="default: pool / expert_hidden")
p.add_argument("--score_func", choices=["sigmoid", "softmax"], default="sigmoid")
p.add_argument("--bias_speed", type=float, default=1e-3)
p.add_argument("--seq_aux", type=float, default=1e-4)
p.add_argument("--sample_window", type=int, default=300, help="steps of probabilistic selection")
p.add_argument("--sample_noise", type=float, default=0.05)
p.add_argument("--transfer_opt", type=int, default=1)
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
LOG_TXT = open(f"logs/{args.run}.log", "w", encoding="utf-8")
LOG_JSON = open(f"logs/{args.run}.jsonl", "w", encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    LOG_TXT.write(msg + "\n")
    LOG_TXT.flush()


def record(**kw):
    LOG_JSON.write(json.dumps(kw) + "\n")
    LOG_JSON.flush()


# ---------------------------------------------------------------- data
meta = json.load(open("data/meta.json"))
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
T = args.block_size
rng = np.random.default_rng(args.data_seed)          # same stream for both phase-2 branches


def batch_from(data, offsets):
    x = np.stack([data[i:i + T + 1] for i in offsets]).astype(np.int64)
    x = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
    return x[:, :-1], x[:, 1:]


def get_train_batch():
    return batch_from(train_data, rng.integers(0, len(train_data) - T - 1, args.batch))


val_offsets = np.random.default_rng(0).integers(0, len(val_data) - T - 1, (args.eval_batches, 32))
amp_dtype = torch.float16 if args.dtype == "fp16" else torch.float32


@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = []
    for offs in val_offsets:
        x, y = batch_from(val_data, offs)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.dtype == "fp16"):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def make_optimizer(model):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=args.lr, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    decay_start = int(args.steps * (1 - args.decay_frac))
    if args.decay_frac > 0 and step >= decay_start:
        frac = (step - decay_start) / max(1, args.steps - decay_start)
        return args.lr * (1 - (1 - args.min_lr_frac) * frac)
    return args.lr


# ---------------------------------------------------------------- model
step0, tokens0 = 0, 0
conversion = None
if args.mode == "dense":
    cfg = ModelConfig(vocab_size=meta["vocab_size"], block_size=T, n_layer=args.n_layer,
                      n_head=args.n_head, n_kv_head=args.n_kv_head, d_model=args.d_model,
                      ffn_hidden=args.ffn_hidden)
    model = GPT(cfg).to(device)
    opt = make_optimizer(model)
else:
    ck = torch.load(args.init, map_location=device, weights_only=False)
    dcfg = ModelConfig(**ck["config"])
    dense = GPT(dcfg).to(device)
    dense.load_state_dict(ck["model"])
    dense_opt = make_optimizer(dense)
    dense_opt.load_state_dict(ck["optimizer"])
    step0, tokens0 = ck["global_step"], ck["tokens"]
    if args.mode == "dense_continue":
        cfg, model, opt = dcfg, dense, dense_opt
    else:
        cfg = ModelConfig(**{**dcfg.to_dict(), "moe": True, "n_experts": args.n_experts,
                             "top_k": args.top_k, "expert_hidden": args.expert_hidden,
                             "shared_hidden": args.shared_hidden, "score_func": args.score_func,
                             "bias_update_speed": args.bias_speed, "seq_aux_alpha": args.seq_aux})
        if args.upcycle == "copy":
            cfg.expert_hidden, cfg.shared_hidden = dcfg.ffn_hidden, 0
        pool = dcfg.ffn_hidden - cfg.shared_hidden
        cfg.route_scale = args.route_scale or (1.0 if args.upcycle == "copy" else pool / cfg.expert_hidden)
        importance = None
        if args.upcycle == "partition" and args.importance:
            crng = np.random.default_rng(123)   # separate rng: keep the training stream identical
            calib = [batch_from(train_data, crng.integers(0, len(train_data) - T - 1, 32))[0]
                     for _ in range(8)]
            importance = up.neuron_importance(dense, calib)
        plan = up.make_plan(dcfg, cfg, args.upcycle, importance, args.drop_ratio, seed=args.seed)
        dense_val = evaluate(dense)
        model = up.upcycle_model(dense, cfg, plan)
        opt = make_optimizer(model)
        moved = up.transfer_adam_state(dense, dense_opt, model, opt, plan) if args.transfer_opt else 0
        moe_val = evaluate(model)
        conversion = {"dense_val_loss": dense_val, "moe_val_loss_at_conversion": moe_val,
                      "adam_tensors_transferred": moved}
        del dense, dense_opt
    for g in opt.param_groups:
        g["lr"] = lr_at(0)

model.train()
scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
total, active = model.param_counts()
tok_per_step = args.batch * args.accum * T
log(f"run={args.run} mode={args.mode} device={device} ({torch.cuda.get_device_name(0)}) dtype={args.dtype}")
log(f"args: {json.dumps(vars(args))}")
log(f"model config: {json.dumps(cfg.to_dict())}")
log(f"params: total {total/1e6:.2f}M  active/token {active/1e6:.2f}M  "
    f"(non-embedding total {(total - cfg.vocab_size*cfg.d_model)/1e6:.2f}M)")
log(f"tokens/step {tok_per_step:,}  starting at global step {step0}, {tokens0/1e6:.1f}M tokens seen")
record(type="config", run=args.run, mode=args.mode, args=vars(args), model=cfg.to_dict(),
       total_params=total, active_params=active, tokens_per_step=tok_per_step,
       step0=step0, tokens0=tokens0, data=meta)
if conversion:
    log(f"conversion: dense val loss {conversion['dense_val_loss']:.4f} -> MoE val loss at step 0 "
        f"{conversion['moe_val_loss_at_conversion']:.4f}  (Adam state tensors carried over: "
        f"{conversion['adam_tensors_transferred']})")
    record(type="conversion", step=0, global_step=step0, tokens=tokens0, **conversion)
    record(type="eval", step=0, global_step=step0, tokens=tokens0,
           val_loss=conversion["moe_val_loss_at_conversion"])
elif args.mode == "dense_continue":
    v = evaluate(model)
    log(f"step {0:5d} | val loss {v:.4f}")
    record(type="eval", step=0, global_step=step0, tokens=tokens0, val_loss=v)

# ---------------------------------------------------------------- train
moe_layers = model.moe_layers()
E = cfg.n_experts if cfg.moe else 0
int_loss, int_aux, int_n = 0.0, 0.0, 0
int_load = torch.zeros(len(moe_layers), max(E, 1), device=device)
int_maxvio = torch.zeros(len(moe_layers), device=device)
t0 = time.time()
t_int = time.time()
best_val = float("inf")
for step in range(1, args.steps + 1):
    lr = lr_at(step - 1)
    for g in opt.param_groups:
        g["lr"] = lr
    noise = args.sample_noise * max(0.0, 1 - (step - 1) / args.sample_window) if args.sample_window else 0.0
    for m in moe_layers:
        m.sample_noise = noise

    loss_sum, aux_sum = 0.0, 0.0
    for _ in range(args.accum):
        x, y = get_train_batch()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.dtype == "fp16"):
            _, loss, aux = model(x, y)
        obj = loss + (aux if aux is not None else 0.0)
        scaler.scale(obj / args.accum).backward()
        loss_sum += loss.item()
        aux_sum += aux.item() if aux is not None else 0.0
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    # aux-loss-free balancing: load counted over the whole (accumulated) batch, then bias update
    for li, m in enumerate(moe_layers):
        load = m.update_bias()
        int_load[li] += load
        int_maxvio[li] += (load.max() - load.mean()) / load.mean()

    int_loss += loss_sum / args.accum
    int_aux += aux_sum / args.accum
    int_n += 1
    gstep, tokens = step0 + step, tokens0 + step * tok_per_step

    if step % args.log_interval == 0 or step == args.steps:
        dt = time.time() - t_int
        rec = dict(type="train", step=step, global_step=gstep, tokens=tokens, loss=int_loss / int_n,
                   lr=lr, grad_norm=gnorm, tok_per_s=int_n * tok_per_step / dt,
                   elapsed_s=time.time() - t0)
        msg = (f"step {step:5d} (global {gstep:5d}, {tokens/1e6:6.1f}M tok) | loss {int_loss/int_n:.4f} "
               f"| lr {lr:.2e} | gnorm {gnorm:.2f} | {rec['tok_per_s']/1e3:.1f}k tok/s")
        if moe_layers:
            frac = int_load / int_load.sum(1, keepdim=True)
            dead = int(((int_load == 0).sum()).item())
            starved = int(((frac < 0.1 / E).sum()).item())
            maxvio_batch = (int_maxvio / int_n).tolist()
            biases = torch.stack([m.expert_bias for m in moe_layers])
            rec.update(aux_loss=int_aux / int_n, sample_noise=noise, load=frac.tolist(),
                       maxvio_batch=maxvio_batch, maxvio_mean=float(np.mean(maxvio_batch)),
                       dead_experts=dead, starved_experts=starved,
                       bias_min=biases.min().item(), bias_max=biases.max().item())
            msg += (f" | aux {int_aux/int_n:.5f} | MaxVio {np.mean(maxvio_batch):.3f} "
                    f"| dead {dead} starved {starved} | bias [{rec['bias_min']:+.3f},{rec['bias_max']:+.3f}]")
            int_load.zero_()
            int_maxvio.zero_()
        log(msg)
        record(**rec)
        int_loss, int_aux, int_n, t_int = 0.0, 0.0, 0, time.time()

    if step % args.eval_interval == 0 or step == args.steps:
        v = evaluate(model)
        best_val = min(best_val, v)
        log(f"step {step:5d} (global {gstep:5d}, {tokens/1e6:6.1f}M tok) | val loss {v:.4f}")
        record(type="eval", step=step, global_step=gstep, tokens=tokens, val_loss=v)

# ---------------------------------------------------------------- save + samples
final_val = evaluate(model)
log(f"final val loss {final_val:.4f}  (training time {(time.time()-t0)/60:.1f} min)")
record(type="final", global_step=step0 + args.steps, tokens=tokens0 + args.steps * tok_per_step,
       val_loss=final_val, train_minutes=(time.time() - t0) / 60)
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
            "global_step": step0 + args.steps, "tokens": tokens0 + args.steps * tok_per_step,
            "val_loss": final_val}, f"checkpoints/{args.run}.pt")

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file("data/tokenizer.json")
model.eval()
torch.manual_seed(42)
with open(f"logs/{args.run}_samples.txt", "w", encoding="utf-8") as f:
    for prompt in ["Once upon a time", "The little dog", "Lily wanted to"]:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        out = model.generate(ids, 120)[0].tolist()
        eos = meta["eos_id"]
        out = out[:out.index(eos, len(ids[0]))] if eos in out[len(ids[0]):] else out
        text = tok.decode(out)
        f.write(f"### prompt: {prompt}\n{text}\n\n")
        log(f"sample [{prompt}]: {text[:300]!r}")
