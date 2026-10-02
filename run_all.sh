#!/usr/bin/env bash
# Full experiment, in order. ~2.5 h on a GTX 1660 Ti (fp32).
set -e
PY=${PY:-python}

# 0. data: TinyStories shard -> 8K BPE -> data/{train,val}.bin
[ -f data/train.bin ] || $PY prepare_data.py

# 1. dense model from scratch: warmup then constant lr (WSD "stable" phase), 2000 steps = 32.8M tokens
$PY train.py --mode dense --run dense --steps 2000 --warmup 200 --eval_interval 100

# 2. how much does each upcycling method disturb the model at the moment of conversion?
$PY compare_upcycling.py --init checkpoints/dense.pt

# 3a. baseline: keep training the dense model (same data stream, same schedule as 3b)
$PY train.py --mode dense_continue --run dense_cont --init checkpoints/dense.pt \
    --steps 2000 --warmup 100 --decay_frac 0.3 --eval_interval 100

# 3b. grow the dense model into a MoE and keep training
#     1 shared expert (512) + 16 routed experts (256), top-2, sigmoid router, route_scale 2,
#     aux-loss-free bias balancing (gamma 1e-3) + sequence aux loss 1e-4, probabilistic
#     selection for the first 300 steps
$PY train.py --mode moe --run moe --init checkpoints/dense.pt \
    --steps 2000 --warmup 100 --decay_frac 0.3 --eval_interval 100

# 4. ablations, 600 steps each at constant lr (comparable with the first 600 steps of 3b)
$PY train.py --mode moe --run moe_nobalance --init checkpoints/dense.pt \
    --steps 600 --warmup 100 --eval_interval 100 --bias_speed 0 --seq_aux 0 --sample_window 0
$PY train.py --mode moe --run moe_scratch_ffn --init checkpoints/dense.pt \
    --steps 600 --warmup 100 --eval_interval 100 --upcycle scratch

# 5. figures and routing analysis
$PY plot.py
$PY analyze_experts.py --ckpt checkpoints/moe.pt
