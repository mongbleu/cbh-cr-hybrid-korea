"""WP-B-5 — box-plots (Prediction vs NIFS CBH) by DBH class and Age class, from the PLOT-DISJOINT table.

Standalone replica of the two box-plot cells in src/paper-Visualize.ipynb (the `compare1_file` data-prep
cell + the grouped-boxplot draw cell). Extracted as a script because the notebook's globals (os.chdir to
another repo, a custom `CBH` module import, CuPy, model loads) make a headless full-notebook run fragile;
the box-plots themselves need only pandas/numpy/matplotlib + the category CSV.

Reads the NEW plot-disjoint category table (from src/wp_b5_category_table.py) and writes TAGGED figures,
leaving the Oct-2025 originals (`fig/boxplot-NIFSvsPred-{AGECLS,DMCLS}.png`) untouched:
  fig/boxplot-NIFSvsPred-DMCLS_plotdisjoint.png
  fig/boxplot-NIFSvsPred-AGECLS_plotdisjoint.png

Prediction box = distribution of per-category `mean` (corrected map); NIFS box = the static `NIFS-CBH`
reference per category (carried forward unchanged). Runs in geo-ml3 or geo-ml4 (no GPU). CBH in meters.

Usage:  python src/wp_b5_boxplots.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                          # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULT_DIR = os.path.join(r"D:\ForestFire\CBH\result\Baseline3", "hybrid-model")
COMPARE_CSV = os.path.join(RESULT_DIR, "pred_vs_NIFS_by_category.csv")
FIG_DIR = r"D:\ForestFire\CBH\fig"
OUT_SUFFIX = "_plotdisjoint"                                    # keep the originals; write tagged figures

# styling — verbatim from the notebook box-plot cell
PRED_FILL, PRED_EDGE = "#A6A6A6", "black"
NIFS_FILL, NIFS_EDGE = "#DCEAF7", "#163E64"
XLABELS = {"DMCLS": "Tree DBH Class", "AGECLS": "Tree Age Class"}


def make_boxplot(df, target_feat):
    """Grouped Prediction-vs-NIFS box-plot for one category axis (DMCLS or AGECLS)."""
    df1 = df[[target_feat, "mean", "NIFS-CBH"]].dropna()
    df1 = df1.loc[df1["NIFS-CBH"] != "-"]
    df1[target_feat] = df1[target_feat].astype("int")
    df1["NIFS-CBH"] = df1["NIFS-CBH"].astype("float")
    ages = sorted(df1[target_feat].unique())

    pred_data, nifs_data = [], []
    for a in ages:
        sub = df1[df1[target_feat] == a]
        pred_data.append(sub["mean"].values)
        nifs_data.append(sub["NIFS-CBH"].values)

    plt.rcParams["font.family"] = "Calibri"
    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(ages); spacing = 3
    centers = np.arange(1, n_groups * spacing + 1, spacing, dtype=float)
    width = 1
    pos_pred = centers - width / 2
    pos_nifs = centers + width / 2

    bp_pred = ax.boxplot(pred_data, positions=pos_pred, widths=0.7, patch_artist=True, showmeans=True,
                         meanprops=dict(marker="x", markersize=6, markeredgecolor=PRED_EDGE),
                         boxprops=dict(color=PRED_EDGE, linewidth=1),
                         whiskerprops=dict(color=PRED_EDGE, linewidth=1),
                         capprops=dict(color=PRED_EDGE, linewidth=1),
                         medianprops=dict(color="black", linewidth=1),
                         flierprops=dict(marker="o", markersize=4, markerfacecolor=PRED_FILL,
                                         markeredgecolor=PRED_EDGE, alpha=0.5))
    bp_nifs = ax.boxplot(nifs_data, positions=pos_nifs, widths=0.7, patch_artist=True, showmeans=True,
                         meanprops=dict(marker="x", markersize=6, markeredgecolor=NIFS_EDGE),
                         boxprops=dict(color=NIFS_EDGE, linewidth=1),
                         whiskerprops=dict(color=NIFS_EDGE, linewidth=1),
                         capprops=dict(linewidth=1),
                         medianprops=dict(color=NIFS_EDGE, linewidth=1),
                         flierprops=dict(marker="o", markersize=4, markerfacecolor=NIFS_EDGE,
                                         markeredgecolor=NIFS_EDGE, alpha=0.5))
    for patch in bp_pred["boxes"]:
        patch.set_facecolor(PRED_FILL)
    for patch in bp_nifs["boxes"]:
        patch.set_facecolor(NIFS_FILL)

    ax.set_xticks(centers)
    ax.set_xticklabels([str(a) for a in ages])
    ax.set_xlabel(XLABELS.get(target_feat, target_feat), fontsize=12, weight="bold")
    ax.set_ylabel("Crown Base Height (m)", fontsize=12, weight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    padding = 1.5
    ax.set_xlim(centers[0] - padding, centers[-1] + padding)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)
    ax.legend(handles=[Patch(facecolor=PRED_FILL, edgecolor="black", label="Prediction (mean)"),
                       Patch(facecolor=NIFS_FILL, edgecolor="black", label="CBH from NIFS")],
              loc="upper left", frameon=True, fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"boxplot-NIFSvsPred-{target_feat}{OUT_SUFFIX}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  ({n_groups} {target_feat} classes, "
          f"{len(df1)} categories with NIFS ref)")


def main():
    if not os.path.exists(COMPARE_CSV):
        raise FileNotFoundError(f"category table not found — run wp_b5_category_table.py first: {COMPARE_CSV}")
    df = pd.read_csv(COMPARE_CSV, encoding="cp949")
    print(f"read {os.path.basename(COMPARE_CSV)}  ({len(df)} rows)")
    for feat in ("DMCLS", "AGECLS"):
        make_boxplot(df, feat)


if __name__ == "__main__":
    main()
