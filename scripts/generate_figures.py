"""
Generate paper-quality figures from eval CSV + failure analysis JSONs.
All data is read from ../results/ and figures are written to ../results/figures/

Usage: python generate_figures.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
FA_DIR = os.path.join(RESULTS_DIR, "failure_analysis")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

COLORS = {
    "baseline": "#3498db",
    "stop_aware_1e7": "#e67e22",
    "stop_aware_5e7": "#27ae60",
    "lost": "#e74c3c",
    "near_miss": "#f39c12",
    "bad_stop": "#95a5a6",
    "success": "#27ae60",
}

# ── Data ────────────────────────────────────────────────────────────────────────

# Per-seed eval data (from terminal output 2026-08-07)
BASELINE_SEEDS = [
    {"seed": 100, "sr": 0.755, "spl": 0.617, "dtg": 1.315, "reward": 7.976},
    {"seed": 200, "sr": 0.885, "spl": 0.712, "dtg": 0.602, "reward": 8.731},
    {"seed": 300, "sr": 0.895, "spl": 0.733, "dtg": 0.612, "reward": 8.962},
]
STOP_AWARE_1E7_SEEDS = [
    {"seed": 100, "sr": 0.895, "spl": 0.730, "dtg": 0.727, "reward": 9.028},
    {"seed": 200, "sr": 0.875, "spl": 0.719, "dtg": 0.770, "reward": 8.808},
    {"seed": 300, "sr": 0.900, "spl": 0.745, "dtg": 0.842, "reward": 8.737},
]
STOP_AWARE_5E7 = {"seed": 300, "sr": 0.945, "spl": 0.820, "dtg": 0.515, "reward": 9.438}

# Failure counts (from failure_analysis JSONs + terminal output)
FAILURE_DATA = {
    "baseline":       {"lost": 58, "near_miss": 30, "bad_stop": 5,  "total": 600, "sr": 0.845},
    "stop_aware_1e7": {"lost": 42, "near_miss": 23, "bad_stop": 1,  "total": 600, "sr": 0.890},
    "stop_aware_5e7": {"lost": 8,  "near_miss": 3,  "bad_stop": 0,  "total": 200, "sr": 0.945},
}


# ── Figure 1: SR bar chart (per-seed + mean) ────────────────────────────────────

def fig1_sr_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.22

    bs = [d["sr"] for d in BASELINE_SEEDS]
    sa = [d["sr"] for d in STOP_AWARE_1E7_SEEDS]
    s5 = [STOP_AWARE_5E7["sr"], None, None]  # single seed at index 2

    bars1 = ax.bar(x - width, bs, width, color=COLORS["baseline"], label="baseline (1e7)", zorder=3)
    bars2 = ax.bar(x, sa, width, color=COLORS["stop_aware_1e7"], label="stop_aware (1e7)", zorder=3)
    bars3 = ax.bar(x[2] + width, STOP_AWARE_5E7["sr"], width, color=COLORS["stop_aware_5e7"],
                   label="stop_aware (5e7)", zorder=3)

    ax.axhline(y=0.845, color=COLORS["baseline"], linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(y=0.890, color=COLORS["stop_aware_1e7"], linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(y=0.945, color=COLORS["stop_aware_5e7"], linestyle="--", alpha=0.6, linewidth=1)

    ax.text(2.4, 0.845, "0.845", color=COLORS["baseline"], va="bottom", fontsize=9, fontweight="bold")
    ax.text(2.4, 0.890, "0.890", color=COLORS["stop_aware_1e7"], va="bottom", fontsize=9, fontweight="bold")
    ax.text(2.4, 0.945, "0.945", color=COLORS["stop_aware_5e7"], va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Seed 100", "Seed 200", "Seed 300"])
    ax.set_ylabel("Success Rate")
    ax.set_title("PointNav HM3D — Val Success Rate")
    ax.legend(loc="lower right")
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_sr_comparison.png"))
    plt.close()
    print("Saved: fig1_sr_comparison.png")


# ── Figure 2: SPL bar chart ────────────────────────────────────────────────────

def fig2_spl_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.22

    bs = [d["spl"] for d in BASELINE_SEEDS]
    sa = [d["spl"] for d in STOP_AWARE_1E7_SEEDS]

    ax.bar(x - width, bs, width, color=COLORS["baseline"], label="baseline (1e7)", zorder=3)
    ax.bar(x, sa, width, color=COLORS["stop_aware_1e7"], label="stop_aware (1e7)", zorder=3)
    ax.bar(x[2] + width, STOP_AWARE_5E7["spl"], width, color=COLORS["stop_aware_5e7"],
           label="stop_aware (5e7)", zorder=3)

    ax.axhline(y=0.687, color=COLORS["baseline"], linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(y=0.731, color=COLORS["stop_aware_1e7"], linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(y=0.820, color=COLORS["stop_aware_5e7"], linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(["Seed 100", "Seed 200", "Seed 300"])
    ax.set_ylabel("SPL")
    ax.set_title("PointNav HM3D — Val SPL")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_spl_comparison.png"))
    plt.close()
    print("Saved: fig2_spl_comparison.png")


# ── Figure 3: Failure analysis stacked bar ──────────────────────────────────────

def fig3_failure_breakdown():
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["lost", "near_miss", "bad_stop"]
    labels = ["Lost (>1m)", "Near-Miss (0.2-0.35m)", "Bad Stop (0.35-1m)"]
    colors = [COLORS["lost"], COLORS["near_miss"], COLORS["bad_stop"]]

    experiments = ["baseline", "stop_aware_1e7", "stop_aware_5e7"]
    exp_labels = ["baseline\n(600 ep)", "stop_aware\n1e7 (600 ep)", "stop_aware\n5e7 (200 ep)"]

    x = np.arange(len(experiments))
    width = 0.5
    bottom = np.zeros(len(experiments))

    for i, cat in enumerate(categories):
        values = [FAILURE_DATA[e][cat] for e in experiments]
        totals = [FAILURE_DATA[e]["total"] for e in experiments]
        pct = [v / t * 100 for v, t in zip(values, totals)]
        bars = ax.bar(x, pct, width, bottom=bottom, color=colors[i], label=labels[i], zorder=3)
        for j, (bar, v) in enumerate(zip(bars, values)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + pct[j]/2,
                        f"n={v}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if colors[i] in ["#e74c3c"] else "black")
        bottom += pct

    # Add SR labels on top
    for j, exp in enumerate(experiments):
        ax.text(j, bottom[j] + 1.5, f"SR={FAILURE_DATA[exp]['sr']:.3f}",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels)
    ax.set_ylabel("% of Episodes")
    ax.set_title("Failure Mode Breakdown")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(bottom) + 6)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_failure_breakdown.png"))
    plt.close()
    print("Saved: fig3_failure_breakdown.png")


# ── Figure 4: DTG distribution ──────────────────────────────────────────────────

def fig4_dtg_distribution():
    """Load DTG values from baseline failure analysis JSONs and plot histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    all_success_dtg = []
    all_fail_dtg = []

    for seed in [100, 200, 300]:
        path = os.path.join(FA_DIR, f"baseline_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for ep in data["episodes"]:
                dtg = ep["distance_to_goal"]
                if ep["success"] > 0.5:
                    all_success_dtg.append(dtg)
                else:
                    all_fail_dtg.append(dtg)

    # Success DTG
    ax1.hist(all_success_dtg, bins=30, color=COLORS["success"], edgecolor="white", alpha=0.85)
    ax1.axvline(x=0.2, color="red", linestyle="--", linewidth=1.5, label="success threshold (0.2m)")
    ax1.set_xlabel("Distance to Goal (m)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Success Episodes (n={len(all_success_dtg)}, "
                  f"mean DTG={np.mean(all_success_dtg):.3f}m)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Failure DTG (log scale for lost vs near_miss separation)
    ax2.hist(all_fail_dtg, bins=40, color=COLORS["lost"], edgecolor="white", alpha=0.85)
    ax2.axvline(x=0.35, color=COLORS["near_miss"], linestyle="--", linewidth=1,
                label="near_miss ≤ 0.35m")
    ax2.axvline(x=1.0, color="gray", linestyle=":", linewidth=1,
                label="bad_stop ≤ 1m")
    ax2.set_xlabel("Distance to Goal (m)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Failure Episodes (n={len(all_fail_dtg)}, "
                  f"near_miss+stop: {sum(1 for d in all_fail_dtg if d <= 1.0)})")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline DTG Distribution (3 seeds × 200ep)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_dtg_distribution.png"))
    plt.close()
    print("Saved: fig4_dtg_distribution.png")


# ── Figure 5: Cross-seed consistency ────────────────────────────────────────────

def fig5_cross_seed_consistency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # SR variation
    bs_sr = [d["sr"] for d in BASELINE_SEEDS]
    sa_sr = [d["sr"] for d in STOP_AWARE_1E7_SEEDS]

    x = [0, 1]
    positions_b = [0 - 0.15] * 3
    positions_s = [1 - 0.15] * 3
    jitter_b = [0 + np.random.uniform(-0.03, 0.03) for _ in range(3)]
    jitter_s = [1 + np.random.uniform(-0.03, 0.03) for _ in range(3)]

    for i in range(3):
        ax1.plot(jitter_b[i], bs_sr[i], "o", color=COLORS["baseline"], markersize=12, zorder=5)
        ax1.plot(jitter_s[i], sa_sr[i], "o", color=COLORS["stop_aware_1e7"], markersize=12, zorder=5)

    ax1.bar(0, np.mean(bs_sr), 0.25, color=COLORS["baseline"], alpha=0.5, zorder=2)
    ax1.bar(1, np.mean(sa_sr), 0.25, color=COLORS["stop_aware_1e7"], alpha=0.5, zorder=2)

    ax1.errorbar(0, np.mean(bs_sr), yerr=np.std(bs_sr), color="black", capsize=5, linewidth=2)
    ax1.errorbar(1, np.mean(sa_sr), yerr=np.std(sa_sr), color="black", capsize=5, linewidth=2)

    ax1.text(0, np.mean(bs_sr) + 0.05, f"std=±{np.std(bs_sr):.3f}", ha="center", fontweight="bold",
             color=COLORS["baseline"])
    ax1.text(1, np.mean(sa_sr) + 0.05, f"std=±{np.std(sa_sr):.3f}", ha="center", fontweight="bold",
             color=COLORS["stop_aware_1e7"])

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["baseline", "stop_aware\n(1e7)"])
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Cross-Seed SR Consistency")
    ax1.set_ylim(0.72, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # SPL variation
    bs_spl = [d["spl"] for d in BASELINE_SEEDS]
    sa_spl = [d["spl"] for d in STOP_AWARE_1E7_SEEDS]

    for i in range(3):
        ax2.plot(jitter_b[i], bs_spl[i], "o", color=COLORS["baseline"], markersize=12, zorder=5)
        ax2.plot(jitter_s[i], sa_spl[i], "o", color=COLORS["stop_aware_1e7"], markersize=12, zorder=5)

    ax2.bar(0, np.mean(bs_spl), 0.25, color=COLORS["baseline"], alpha=0.5, zorder=2)
    ax2.bar(1, np.mean(sa_spl), 0.25, color=COLORS["stop_aware_1e7"], alpha=0.5, zorder=2)

    ax2.errorbar(0, np.mean(bs_spl), yerr=np.std(bs_spl), color="black", capsize=5, linewidth=2)
    ax2.errorbar(1, np.mean(sa_spl), yerr=np.std(sa_spl), color="black", capsize=5, linewidth=2)

    ax2.text(0, np.mean(bs_spl) + 0.04, f"std=±{np.std(bs_spl):.3f}", ha="center", fontweight="bold",
             color=COLORS["baseline"])
    ax2.text(1, np.mean(sa_spl) + 0.04, f"std=±{np.std(sa_spl):.3f}", ha="center", fontweight="bold",
             color=COLORS["stop_aware_1e7"])

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["baseline", "stop_aware\n(1e7)"])
    ax2.set_ylabel("SPL")
    ax2.set_title("Cross-Seed SPL Consistency")
    ax2.set_ylim(0.55, 0.85)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Training Consistency Improvement with Stop-Aware Reward",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_cross_seed_consistency.png"))
    plt.close()
    print("Saved: fig5_cross_seed_consistency.png")


# ── Figure 6: Improvement trajectory (SR + SPL path) ─────────────────────────────

def fig6_improvement_path():
    fig, ax1 = plt.subplots(figsize=(7, 5))

    stages = ["baseline\n(1e7)", "stop_aware\n(1e7)", "stop_aware\n(5e7)"]
    sr = [0.845, 0.890, 0.945]
    spl = [0.687, 0.731, 0.820]

    ax1.plot([0, 1, 2], sr, "o-", color=COLORS["stop_aware_5e7"], linewidth=2.5,
             markersize=14, label="Success Rate", zorder=5)
    ax2 = ax1.twinx()
    ax2.plot([0, 1, 2], spl, "s--", color=COLORS["stop_aware_1e7"], linewidth=2.5,
             markersize=14, label="SPL", zorder=5)

    for i, (s, p) in enumerate(zip(sr, spl)):
        ax1.annotate(f"{s:.3f}", (i, s), textcoords="offset points", xytext=(15, 10),
                     fontsize=12, fontweight="bold", color=COLORS["stop_aware_5e7"])
        ax2.annotate(f"{p:.3f}", (i, p), textcoords="offset points", xytext=(15, -15),
                     fontsize=12, fontweight="bold", color=COLORS["stop_aware_1e7"])

    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(stages)
    ax1.set_ylabel("Success Rate", color=COLORS["stop_aware_5e7"])
    ax2.set_ylabel("SPL", color=COLORS["stop_aware_1e7"])
    ax1.tick_params(axis="y", labelcolor=COLORS["stop_aware_5e7"])
    ax2.tick_params(axis="y", labelcolor=COLORS["stop_aware_1e7"])
    ax1.set_ylim(0.80, 1.00)
    ax2.set_ylim(0.60, 0.88)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("Improvement Trajectory")
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_improvement_path.png"))
    plt.close()
    print("Saved: fig6_improvement_path.png")


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig1_sr_comparison()
    fig2_spl_comparison()
    fig3_failure_breakdown()
    fig4_dtg_distribution()
    fig5_cross_seed_consistency()
    fig6_improvement_path()
    print(f"\nDone. {len(os.listdir(FIG_DIR))} figures in {FIG_DIR}")
