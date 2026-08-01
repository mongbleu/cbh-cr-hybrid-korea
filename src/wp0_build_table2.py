"""WP-0 — assemble the corrected Table 2 from the plot-disjoint re-train outputs.

Merges baseline + hybrid, tree + plot per-species CSVs into one manuscript-ready table with:
  - integer per-species n (fixes R2-10's n=0.594 column/merge bug),
  - plot-disjoint R²/RMSE/MAE/bias for baseline and hybrid (the reported values),
  - the tree-vs-plot R² delta (the leakage effect, R2-3),
  - a pooled/median summary row.
Read-only on inputs; writes Table2_model_performance.csv next to the eval CSVs.
"""
import os, glob
import pandas as pd

DIR = glob.glob("result/Baseline3/*plotdisjoint")[0]            # the tagged run dir
SEED = 100


def _load(model, mode):                                         # read one eval CSV
    return pd.read_csv(os.path.join(DIR, f"{model}_eval_{mode}_SEED{SEED}.csv"))


def main():
    bt, bp = _load("baseline", "tree"), _load("baseline", "plot")   # baseline tree/plot
    ht, hp = _load("hybrid", "tree"), _load("hybrid", "plot")       # hybrid tree/plot

    # Per-species total n (integer) straight from the cleaned data — the honest count (R2-10).
    df = None
    for enc in ("cp949", "utf-8"):
        try:
            df = pd.read_csv("data/NFI6-7_cleaned2.csv", encoding=enc); break
        except Exception:
            continue
    n_tot = df.dropna(subset=["SID", "CR"]).groupby("SID").size().rename("n_total").astype(int)

    # Baseline plot-disjoint (reported) + tree R² for the delta.
    base = bp[["SID", "n_test", "R2_Test", "RMSE_Test", "MAE_Test", "Bias_Test"]].rename(
        columns={"n_test": "n_test_plot", "R2_Test": "base_R2_plot", "RMSE_Test": "base_RMSE_plot",
                 "MAE_Test": "base_MAE_plot", "Bias_Test": "base_Bias_plot"})
    base = base.merge(bt[["SID", "R2_Test"]].rename(columns={"R2_Test": "base_R2_tree"}), on="SID", how="left")

    # Hybrid plot-disjoint (reported) + tree R² for the delta.
    hyb = hp[["SID", "R2_Test", "RMSE_Test", "MAE_Test", "Bias_Test"]].rename(
        columns={"R2_Test": "hyb_R2_plot", "RMSE_Test": "hyb_RMSE_plot",
                 "MAE_Test": "hyb_MAE_plot", "Bias_Test": "hyb_Bias_plot"})
    hyb = hyb.merge(ht[["SID", "R2_Test"]].rename(columns={"R2_Test": "hyb_R2_tree"}), on="SID", how="left")

    t = base.merge(hyb, on="SID", how="outer").merge(n_tot, on="SID", how="left")
    t["base_R2_delta"] = (t["base_R2_tree"] - t["base_R2_plot"]).round(4)   # leakage effect (baseline)
    t["hyb_R2_delta"] = (t["hyb_R2_tree"] - t["hyb_R2_plot"]).round(4)      # leakage effect (hybrid)
    t["n_total"] = t["n_total"].astype("Int64")                            # integer n (R2-10)
    t = t.sort_values("SID").reset_index(drop=True)

    cols = ["SID", "n_total", "n_test_plot",
            "base_R2_plot", "base_RMSE_plot", "base_MAE_plot", "base_Bias_plot", "base_R2_tree", "base_R2_delta",
            "hyb_R2_plot", "hyb_RMSE_plot", "hyb_MAE_plot", "hyb_Bias_plot", "hyb_R2_tree", "hyb_R2_delta"]
    t = t[cols]
    out = os.path.join(DIR, "Table2_model_performance.csv")
    t.to_csv(out, index=False, encoding="utf-8-sig")

    # Median summary (R² is skewed; median is the honest center).
    s = {"hyb_R2_plot_median": t.hyb_R2_plot.median(), "hyb_R2_tree_median": t.hyb_R2_tree.median(),
         "base_R2_plot_median": t.base_R2_plot.median(), "base_R2_tree_median": t.base_R2_tree.median(),
         "hyb_species_R2le0_plot": int((t.hyb_R2_plot <= 0).sum()),
         "hyb_species_R2le0_tree": int((t.hyb_R2_tree <= 0).sum()), "species": len(t)}
    print("Table 2 written:", out, f"({len(t)} species)")
    print("all n_total integer:", bool(t.n_total.dropna().apply(float.is_integer if False else (lambda v: float(v).is_integer())).all()))
    print("medians  hybrid: tree {hyb_R2_tree_median:.3f} -> plot {hyb_R2_plot_median:.3f} | "
          "baseline: tree {base_R2_tree_median:.3f} -> plot {base_R2_plot_median:.3f}".format(**s))
    print("hybrid species R2<=0: tree {hyb_species_R2le0_tree} -> plot {hyb_species_R2le0_plot} / {species}".format(**s))


if __name__ == "__main__":
    main()
