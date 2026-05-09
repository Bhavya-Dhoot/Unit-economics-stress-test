"""
Visualization module — publication-quality experiment plots.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from config import RESULTS_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                      "axes.titleweight": "bold", "axes.labelweight": "bold"})

_PAL = {"signal_A": "#4C72B0", "signal_B": "#DD8452", "signal_C": "#55A868"}


def plot_fragility_by_source(df_with_bins, decomp_summary, output_dir=RESULTS_DIR):
    output_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = decomp_summary["cac_bin"].tolist()
    sns.boxplot(data=df_with_bins, x="cac_bin", y="rank_delta",
                order=order, hue="cac_bin", palette="coolwarm",
                showfliers=False, legend=False, ax=axes[0])
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Rank Delta Distribution by CAC Source")
    axes[0].set_xlabel("CAC Source Bin"); axes[0].set_ylabel("Rank Delta")
    axes[0].tick_params(axis="x", rotation=15)
    colors = sns.color_palette("coolwarm", n_colors=len(decomp_summary))
    axes[1].bar(decomp_summary["cac_bin"].astype(str),
                decomp_summary["inversion_rate"], color=colors,
                edgecolor="black", linewidth=0.5)
    axes[1].set_title("Inversion Rate by CAC Source Bin")
    axes[1].set_ylabel("Fraction Inverted")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
    axes[1].tick_params(axis="x", rotation=15)
    # Proportional annotation offset (5% of max bar height, min 0.0001)
    max_inv = max(decomp_summary["inversion_rate"].max(), 0.0001)
    offset = max_inv * 0.08
    for i, (_, row) in enumerate(decomp_summary.iterrows()):
        axes[1].text(i, row["inversion_rate"] + offset,
                     f"{row['inversion_rate']:.2%}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("Hypothesis 1: Decomposition Fragility Under 30% CAC Shock",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fragility_by_source.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}"); return path


def plot_signal_comparison(signal_df, output_dir=RESULTS_DIR):
    output_dir.mkdir(exist_ok=True)
    overall = signal_df[signal_df["scope"] == "overall"].copy()
    if overall.empty: return output_dir / "signal_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    signals = overall["signal"].tolist()
    colors = [_PAL.get(s, "#999") for s in signals]
    lbls = {"signal_A": "A: Static", "signal_B": "B: Trend", "signal_C": "C: Composite"}
    xl = [lbls.get(s, s) for s in signals]
    b1 = axes[0].bar(xl, overall["spearman_rho"].values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Spearman ρ with Stressed Rank"); axes[0].set_ylim(0, 1)
    for bar, v in zip(b1, overall["spearman_rho"].values):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    b2 = axes[1].bar(xl, overall["top_quartile_precision"].values, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Top-Quartile Precision"); axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    for bar, v in zip(b2, overall["top_quartile_precision"].values):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.1%}", ha="center", fontweight="bold")
    fig.suptitle("Hypothesis 2: Signal Predictive Power Under Stress", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "signal_comparison.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}"); return path


def plot_sector_heatmap(sector_df, output_dir=RESULTS_DIR):
    output_dir.mkdir(exist_ok=True)
    if sector_df.empty: return output_dir / "sector_heatmap.png"
    cols = ["prec_A", "prec_C", "precision_gap"]
    if "prec_B" in sector_df.columns: cols = ["prec_A", "prec_B", "prec_C", "precision_gap"]
    pivot = sector_df.set_index("scope")[cols]
    pivot.columns = [c.replace("prec_", "Signal ").replace("precision_gap", "Gap (C-A)") for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, max(5, len(pivot)*0.7+1)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Signal Precision by Sector", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "sector_heatmap.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}"); return path


def plot_rank_delta_distribution(df, output_dir=RESULTS_DIR):
    output_dir.mkdir(exist_ok=True)
    if "cac_bin" not in df.columns: return output_dir / "rank_delta_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    cac = df[df["cac_bin"] == "CAC-Driven (75-100%)"]["rank_delta"].dropna()
    ltv = df[df["cac_bin"] == "LTV-Driven (0-25%)"]["rank_delta"].dropna()
    if len(cac) == 0:
        cac = df[df["cac_bin"].astype(str).str.contains("CAC")]["rank_delta"].dropna()
    if len(ltv) == 0:
        ltv = df[df["cac_bin"].astype(str).str.contains("LTV")]["rank_delta"].dropna()
    ax.hist(ltv, bins=80, alpha=0.6, label="LTV-Driven", color="#55A868", density=True)
    ax.hist(cac, bins=80, alpha=0.6, label="CAC-Driven", color="#C44E52", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Rank Delta: LTV-Driven vs CAC-Driven", fontweight="bold")
    ax.set_xlabel("Rank Delta"); ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    path = output_dir / "rank_delta_distribution.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}"); return path
