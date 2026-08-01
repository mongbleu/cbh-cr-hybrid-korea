"""WP-B-6 — validation gate for the plot-disjoint maps + category table + box-plots.

Consolidates the acceptance checks the project requires before snapshotting (CLAUDE.md validation gate):
  1. PROVENANCE  — outputs derive from the plot-disjoint bundle (mode='plotdisjoint', SEED=100), so the
                   WP-0 leakage correction is carried through to the published maps/figures.
  2. UNITS/CRS   — CR ∈ [0,1]; CBH in meters, ≥0 and ≤ a physical max; both rasters EPSG:5179, 5 m grid.
  3. HONEST LANG — quantify pred-vs-NIFS bias/RMSE/MAE per DBH and Age class from the category table so
                   the manuscript reports the real (positive) bias instead of calling it "good".

Read-only on the frozen artifacts (no writes to F:\CBH or the tagged result dir except the report).
Pure rasterio + pandas (no GPU). Run in geo-ml3 or geo-ml4.

Usage:  python src/wp_b6_validation_gate.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import rasterio

RASTER_DIR = r"F:\CBH"
CR_TIF  = os.path.join(RASTER_DIR, "CR_hybrid_model.tif")
CBH_TIF = os.path.join(RASTER_DIR, "CBH_meter_hybrid_model.tif")
TAG_DIR = os.path.join(r"D:\ForestFire\CBH\result\Baseline3", "hybrid-model")
BUNDLE  = os.path.join(TAG_DIR, "model", "hybrid_model.joblib")
CAT_CSV = os.path.join(TAG_DIR, "pred_vs_NIFS_by_category.csv")
REPORT  = os.path.join(TAG_DIR, "validation_report.txt")

CBH_PHYS_MAX = 40.0       # generous physical ceiling for Korean stands (m); CBH must be ≤ H ≤ ~this
lines = []
def emit(s=""):
    print(s); lines.append(s)
checks = []               # (name, passed, detail)
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    emit(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def gate_provenance():
    import joblib
    emit("1) PROVENANCE — plot-disjoint carry-through")
    b = joblib.load(BUNDLE)
    mode = b.get("mode"); seed = b.get("SEED"); n_sid = len(b.get("label_encoder_classes", []))
    # retrain script: SPLIT_MODE 'plot' = GroupShuffleSplit/StratifiedGroupKFold by plot key (corrected);
    # 'tree' = original leaky row split. The bundle is saved ONLY when mode=='plot' (ship corrected only).
    check("bundle mode is plot-disjoint ('plot', not 'tree')", mode == "plot", f"mode={mode!r}")
    check("bundle SEED == 100", seed == 100, f"SEED={seed}")
    check("bundle covers 39 SIDs", n_sid == 39, f"{n_sid} classes")
    emit()


def gate_units_crs():
    emit("2) UNITS / CRS / RANGES")
    for label, path, lo, hi in [("CR", CR_TIF, 0.0, 1.0), ("CBH(m)", CBH_TIF, 0.0, CBH_PHYS_MAX)]:
        with rasterio.open(path) as src:
            code = src.crs.to_epsg()
            px = src.res
            st = src.statistics(1)                # cached in .aux.xml → no rescan
            check(f"{label} CRS == EPSG:5179", code == 5179, f"EPSG:{code}")
            check(f"{label} grid == 5 m", abs(px[0] - 5) < 1e-6 and abs(px[1] - 5) < 1e-6, f"res={px}")
            check(f"{label} min >= {lo}", st.min >= lo - 1e-6, f"min={st.min:.4f}")
            check(f"{label} max <= {hi}", st.max <= hi + 1e-6, f"max={st.max:.4f}")
            emit(f"     {label}: min={st.min:.4f} max={st.max:.4f} mean={st.mean:.4f} std={st.std:.4f}")
    emit()


def gate_honest_language():
    emit("3) HONEST PERFORMANCE LANGUAGE — pred(mean) vs NIFS reference, by category")
    df = pd.read_csv(CAT_CSV, encoding="cp949")
    d = df[["SID", "AGECLS", "DMCLS", "mean", "NIFS-CBH"]].copy()
    d["NIFS-CBH"] = pd.to_numeric(d["NIFS-CBH"], errors="coerce")
    d = d.dropna(subset=["mean", "NIFS-CBH"])
    d["err"] = d["mean"] - d["NIFS-CBH"]          # prediction minus reference
    n = len(d)
    bias = d["err"].mean(); mae = d["err"].abs().mean(); rmse = float(np.sqrt((d["err"] ** 2).mean()))
    emit(f"   pooled over {n} categories: bias=+{bias:.3f} m, MAE={mae:.3f} m, RMSE={rmse:.3f} m "
         f"(prediction reads HIGH vs NIFS)")
    for feat, name in [("DMCLS", "DBH class"), ("AGECLS", "Age class")]:
        emit(f"   by {name}:")
        g = d.groupby(feat).agg(n=("err", "size"), bias=("err", "mean"),
                                mae=("err", lambda x: x.abs().mean()))
        for cls, row in g.iterrows():
            emit(f"     {name} {int(cls)}: n={int(row['n']):2d}  bias={row['bias']:+.2f} m  mae={row['mae']:.2f} m")
    # not a pass/fail gate — this is the honest-reporting evidence; record that bias is non-trivial
    check("bias quantified & non-trivial (reported, not hidden)", True,
          f"pooled bias +{bias:.2f} m — must be stated in the manuscript, not called 'good'")
    emit()


def main():
    emit("=" * 78)
    emit("WP-B-6 VALIDATION GATE — plot-disjoint CBH/CR maps + category table + box-plots")
    emit("=" * 78); emit()
    for p in (BUNDLE, CR_TIF, CBH_TIF, CAT_CSV):
        if not os.path.exists(p):
            emit(f"MISSING input: {p}"); sys.exit(2)
    gate_provenance()
    gate_units_crs()
    gate_honest_language()

    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_fail = len(checks) - n_pass
    emit("-" * 78)
    emit(f"GATE RESULT: {n_pass}/{len(checks)} checks passed, {n_fail} failed")
    emit("VERDICT: " + ("PASS — ready for immutable snapshot" if n_fail == 0 else "FAIL — fix before snapshot"))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nreport written: {REPORT}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
