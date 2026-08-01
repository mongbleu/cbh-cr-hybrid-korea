"""Per-class Hybrid-vs-NIFS CBH bias (cell-equal-weighted), matching the Figure 5 aggregation.

Reads the plot-disjoint category table and differences the prediction `mean` against the static
`NIFS-CBH` reference per (SID,AGECLS,DMCLS) cell, then summarizes bias by DBH class and Age class.
Cell-equal-weighted (one point per category cell) = same basis as the Figure 5 box-plots.
Read-only analysis; writes a small summary CSV. CBH in meters.
"""
import os
import pandas as pd

P = r"result/Baseline3/hybrid-model/pred_vs_NIFS_by_category.csv"
OUT = r"result/Baseline3/hybrid-model/perclass_bias.csv"

df = pd.read_csv(P, encoding="cp949")
df = df[df["NIFS-CBH"] != "-"].copy()
df["NIFS-CBH"] = pd.to_numeric(df["NIFS-CBH"], errors="coerce")
df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
df = df.dropna(subset=["mean", "NIFS-CBH"])
df["bias"] = df["mean"] - df["NIFS-CBH"]

print("cells with NIFS ref:", len(df))
print("POOLED (cell-equal-weight): bias mean=%.3f median=%.3f MAE=%.3f"
      % (df.bias.mean(), df.bias.median(), df.bias.abs().mean()))

rows = []
for feat in ("DMCLS", "AGECLS"):
    g = df.groupby(feat)["bias"].agg(n_cells="count", bias_mean="mean",
                                     bias_median="median", MAE=lambda s: s.abs().mean())
    print("\n--- by %s ---" % feat)
    print(g.round(3).to_string())
    gg = g.reset_index().rename(columns={feat: "class_value"})
    gg.insert(0, "class_axis", feat)
    rows.append(gg)

summ = pd.concat(rows, ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
summ.round(4).to_csv(OUT, index=False, encoding="utf-8-sig")
print("\nwrote", OUT)
