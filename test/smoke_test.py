"""Smoke test for the CBH-CR pipeline — fast, data-light, reproducibility check.

Validates that the cleaned modeling table and the plot-disjoint pipeline work end to end, without the
multi-GB rasters. It exercises the REAL code in src/wp0_plot_disjoint_retrain.py (schema, units, plot
key, GroupShuffleSplit disjointness, baseline fit). Because the NFI-derived data is NOT redistributable,
the test SKIPS (exit 0) when the cleaned table is absent rather than failing.

Run (in the geo-ml3 env):  python test/smoke_test.py
Exit codes: 0 = pass or skipped (data absent); 1 = failure.
"""
import os                                                       # env + paths
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")           # avoid the geo-ml4 OpenMP curve_fit crash
os.environ.setdefault("OMP_NUM_THREADS", "1")                   # (harmless in geo-ml3)
import sys, importlib.util                                      # module loading + exit codes
import numpy as np, pandas as pd                                # data + math

HERE = os.path.dirname(os.path.abspath(__file__))               # test/ dir
ROOT = os.path.dirname(HERE)                                    # repo root
SCRIPT = os.path.join(ROOT, "src", "wp0_plot_disjoint_retrain.py")  # the code under test
CANDIDATES = [os.path.join(HERE, "data", "NFI-Integrated(6-7)-cleaned.csv"),  # preferred (test slice)
              os.path.join(ROOT, "data", "NFI6-7_cleaned2.csv")]              # fallback (full data)
REQUIRED = ["SID", "SampleID", "Cycle", "DBH(inch)", "H(ft)", "CR"]  # minimal modeling columns


def load_module():                                              # import the pipeline code as a module
    spec = importlib.util.spec_from_file_location("wp0", SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def find_data():                                                # first cleaned table that exists
    return next((p for p in CANDIDATES if os.path.exists(p)), None)


def read(path):                                                 # cp949 then utf-8
    for enc in ("cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"could not read {path}")


def main():
    if not os.path.exists(SCRIPT):                              # the code itself must be present
        print("FAIL: missing", SCRIPT); return 1
    path = find_data()
    if path is None:                                            # data absent → SKIP (not a failure)
        print("SKIP: cleaned NFI table not found (not redistributable). See test/data/data-requirements.txt")
        print("      expected one of:\n        - " + "\n        - ".join(CANDIDATES))
        return 0

    mod = load_module()                                         # load pipeline code
    df = read(path)
    checks = []                                                 # (name, ok) pairs

    miss = [c for c in REQUIRED if c not in df.columns]         # 1) schema
    checks.append(("schema: required columns present", not miss))
    df = df.dropna(subset=REQUIRED)

    checks.append(("units: CR within [0,1]", bool(df["CR"].between(0, 1).all())))  # 2) units

    g = df["SampleID"]                                          # 3) plot key has >1 tree/plot
    tpp = len(df) / max(g.nunique(), 1)
    checks.append((f"plot key: {g.nunique()} plots, {tpp:.1f} trees/plot (>1)", tpp > 1))

    # 4) plot-disjoint split + baseline fit on a few well-sampled species (real code path)
    sids = (df.groupby("SID").size().sort_values(ascending=False).index[:3])  # 3 largest species
    fit_ok = split_ok = True
    for sid in sids:
        dfx = df[df["SID"] == sid]
        try:
            train, test = mod.split_species(dfx, "plot", None)  # GroupShuffleSplit + internal disjoint assert
            if not set(train["SampleID"]).isdisjoint(set(test["SampleID"])):
                split_ok = False                                # belt-and-suspenders re-check
            params = mod.fit_baseline(train, "plot")            # fit HM baseline on training plots
            m = mod.evaluate(params, test)                      # held-out metrics
            if not np.isfinite(m["R2"]):                        # must produce a finite score
                fit_ok = False
        except Exception as e:
            print("   exception for SID", int(sid), ":", e); fit_ok = False
    checks.append(("split: train/test plots disjoint (3 species)", split_ok))
    checks.append(("baseline: fits + finite R² on held-out plots", fit_ok))

    print(f"Smoke test on {os.path.relpath(path, ROOT)}  (rows={len(df)})")
    ok_all = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"); ok_all &= ok
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
