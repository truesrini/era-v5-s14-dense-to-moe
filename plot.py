"""
Read logs/*.jsonl and draw the figures used in the README into plots/.

    python plot.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

RUNS = {  # run name -> (label, colour)
    "dense": ("Dense, phase 1", BLUE),
    "dense_cont": ("Dense, kept training (baseline)", BLUE),
    "moe": ("MoE, upcycled from dense", ORANGE),
    "moe_nobalance": ("MoE, no balancing (ablation)", AQUA),
    "moe_scratch_ffn": ("MoE, FFN from scratch (ablation)", VIOLET),
}


def load(run):
    path = f"logs/{run}.jsonl"
    if not os.path.exists(path):
        return None
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {t: [r for r in recs if r["type"] == t] for t in ("config", "train", "eval", "conversion", "final")}


def ema(x, a=0.9):
    out, m = [], None
    for v in x:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


logs = {r: load(r) for r in RUNS}
logs = {r: v for r, v in logs.items() if v}
os.makedirs("plots", exist_ok=True)
conv_tok = logs["moe"]["config"][0]["tokens0"] / 1e6 if "moe" in logs else None


def end_label(ax, x, y, text, left=False):
    ax.annotate(text, (x, y), xytext=(-8 if left else 6, 0), textcoords="offset points", va="center",
                ha="right" if left else "left", color=INK2, fontsize=9)


def mark_conversion(ax):
    if conv_tok is not None:
        ax.axvline(conv_tok, color=INK2, linewidth=1)
        ax.annotate("dense → MoE", (conv_tok, 0), xycoords=("data", "axes fraction"),
                    xytext=(-4, 4), textcoords="offset points", va="bottom", ha="right",
                    color=INK2, fontsize=9)


# ---------------------------------------------------------------- 1. validation loss, whole run
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
for run in ["dense", "dense_cont", "moe", "moe_nobalance", "moe_scratch_ffn"]:
    if run not in logs:
        continue
    ev = logs[run]["eval"]
    x = [e["tokens"] / 1e6 for e in ev]
    y = [e["val_loss"] for e in ev]
    label, col = RUNS[run]
    ax.plot(x, y, color=col, label=label if run != "dense" else None,
            linestyle="-" if run in ("dense", "dense_cont", "moe") else (0, (4, 2)))
ax.set_title("Validation loss over the whole run")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("validation loss (nats/token)")
mark_conversion(ax)
ax.legend(loc="upper right")
lo = min(e["val_loss"] for r in logs.values() for e in r["eval"])
ax.set_ylim(lo - 0.05, lo + 1.2)

ax = axes[1]                                                # zoom on phase 2
for run in ["dense_cont", "moe", "moe_nobalance", "moe_scratch_ffn"]:
    if run not in logs:
        continue
    ev = logs[run]["eval"]
    x = [e["tokens"] / 1e6 for e in ev]
    y = [e["val_loss"] for e in ev]
    label, col = RUNS[run]
    ax.plot(x, y, color=col, linestyle="-" if run in ("dense_cont", "moe") else (0, (4, 2)))
    ax.plot(x[-1], y[-1], "o", color=col, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    end_label(ax, x[-1], y[-1], f"{y[-1]:.3f}")
if "dense" in logs:
    d = logs["dense"]["eval"][-1]
    ax.plot(d["tokens"] / 1e6, d["val_loss"], "o", color=BLUE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    end_label(ax, d["tokens"] / 1e6, d["val_loss"], f"dense {d['val_loss']:.3f}", left=True)
if "moe" in logs:
    c = logs["moe"]["conversion"][0]
    ax.plot(c["tokens"] / 1e6, c["moe_val_loss_at_conversion"], "o", color=ORANGE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    end_label(ax, c["tokens"] / 1e6, c["moe_val_loss_at_conversion"],
              f"MoE {c['moe_val_loss_at_conversion']:.3f}", left=True)
    ax.set_xlim(conv_tok - 9, None)
ax.set_title("After conversion, all branches from one checkpoint")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("validation loss (nats/token)")
ph2 = [e["val_loss"] for r in ("dense_cont", "moe") if r in logs for e in logs[r]["eval"]]
if ph2:
    ax.set_ylim(min(ph2) - 0.03, max(ph2) + 0.05)
fig.tight_layout()
fig.savefig("plots/val_loss.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 2. training loss
fig, ax = plt.subplots(figsize=(13, 4.2))
for run in ["dense", "dense_cont", "moe", "moe_nobalance", "moe_scratch_ffn"]:
    if run not in logs:
        continue
    tr = logs[run]["train"]
    x = np.array([t["tokens"] / 1e6 for t in tr])
    y = ema([t["loss"] for t in tr])
    label, col = RUNS[run]
    ax.plot(x, y, color=col, label=label if run != "dense" else None,
            linestyle="-" if run in ("dense", "dense_cont", "moe") else (0, (4, 2)))
ax.set_title("Training loss (smoothed)")
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("training loss (nats/token)")
lo = min(t["loss"] for r in logs.values() for t in r["train"])
ax.set_ylim(lo - 0.05, lo + 1.2)
mark_conversion(ax)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("plots/train_loss.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------- 3. expert load heatmaps
def load_heatmap(run, fname, title):
    tr = [t for t in logs[run]["train"] if "load" in t]
    load = np.array([t["load"] for t in tr])                       # (logs, layers, experts)
    steps = [t["step"] for t in tr]
    L, E = load.shape[1], load.shape[2]
    fig, axes = plt.subplots(1, L, figsize=(2.3 * L, 4.2), sharey=True)
    vmax = max(3.0 / E, load.max())
    for li in range(L):
        ax = axes[li]
        im = ax.imshow(load[:, li, :], aspect="auto", cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest",
                       extent=[-0.5, E - 0.5, steps[-1], steps[0]])
        ax.set_title(f"layer {li}", fontsize=10)
        ax.set_xlabel("expert")
        ax.grid(False)
        ax.set_xticks([0, E // 2, E - 1])
    axes[0].set_ylabel("MoE training step")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(f"share of routed tokens (even share = {1/E:.4f})")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


if "moe" in logs:
    load_heatmap("moe", "plots/expert_load_moe.png",
                 "Expert load per layer, upcycled MoE with bias balancing")
if "moe_nobalance" in logs:
    load_heatmap("moe_nobalance", "plots/expert_load_nobalance.png",
                 "Expert load per layer, no balancing (ablation)")

# ---------------------------------------------------------------- 4. balance metrics (separate charts, no dual axis)
bal_runs = [r for r in ("moe", "moe_nobalance") if r in logs]
if bal_runs:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for run in bal_runs:
        tr = [t for t in logs[run]["train"] if "maxvio_mean" in t]
        x = [t["step"] for t in tr]
        label, col = RUNS[run]
        axes[0].plot(x, [t["maxvio_mean"] for t in tr], color=col, label=label)
        axes[1].plot(x, [t["starved_experts"] for t in tr], color=col, label=label)
    axes[0].set_title("Load imbalance: MaxVio per batch, mean over layers")
    axes[0].set_ylabel("MaxVio  (0 = perfectly even)")
    axes[1].set_title("Starved experts (< 10% of an even share), all layers")
    axes[1].set_ylabel("experts")
    for ax in axes:
        ax.set_xlabel("MoE training step")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("plots/balance.png", dpi=150)
    plt.close(fig)

print("wrote", sorted(os.listdir("plots")))
