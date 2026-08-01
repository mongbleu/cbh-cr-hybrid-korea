"""WP-B-3 dry-run: prove the NEW (plot-disjoint bundle) map prediction logic on synthetic rows.

NON-GPU, no CuPy / rasterio / heavy I/O. Replicates the exact code path that CRModel3-Mapping.ipynb
CELL 31 (bundle load) + CELL 41 (prediction) now use, in plain numpy/pandas, and asserts:
  - in-bundle SID rows  -> CR_pred in [0,1], CBH=(1-CR)*H finite and 0 <= CBH <= H;
  - SID=77 (mixed forest, out-of-bundle) -> MASKED (NaN), never passed through the encoder.
Also cross-checks that the map-time baseline CR_pred reproduces wp0_plot_disjoint_retrain.func4
bit-for-bit (same func, same [DBH(inch), H(ft)] feature order).

Run (geo-ml3):
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  C:/Users/user/anaconda3/envs/geo-ml3/python.exe test/wp_b3_dryrun.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

ROOT = r"D:\ForestFire\CBH"
sys.path.insert(0, os.path.join(ROOT, "src"))
from wp0_plot_disjoint_retrain import func4 as func4_ref  # the authoritative training-time baseline

BUNDLE = os.path.join(ROOT, "result", "Baseline3",
                      "hybrid-model", "model",
                      "hybrid_model.joblib")

b = joblib.load(BUNDLE)
XGB_MODEL    = b["xgb_model"]
SCALER       = b["scaler"]
PARAMS_BY_SID = b["params_by_sid"]
FEATURE_COLS = list(b["FEATURE_COLS"])
NUMERIC_COLS = list(b["NUMERIC_COLS"])
LE_CLASSES   = np.asarray(b["label_encoder_classes"], dtype=np.int64)
print(f"bundle: {len(LE_CLASSES)} SIDs, SEED={b['SEED']}, mode={b['mode']}")
print(f"FEATURE_COLS={FEATURE_COLS}")


# --- replicate CELL 31's func4 exactly (column order [DBH(inch), H(ft)]) ---
def func4_map(X, a1, a2, a3, b_):
    H, D = X
    H_log, D_log = np.log1p(H), np.log1p(D)
    z = (a1 * H_log / D_log) + (a2 * H_log) + (a3 * D_log ** 2) + b_
    return 1.0 / (1.0 + np.exp(-z))


# --- the NEW map prediction path, replicated on CPU exactly as CELL 41 does it ---
def predict_block(rows):
    """rows: list of dicts with raw pixel quantities (already in map-time units:
       H(ft), DBH(inch), CD(%), Elev(hm), Slope(tan), Azimuth(rad), Lat, Long, and raw KOFTR SID).
    Returns DataFrame with CR and CBH (NaN where masked). CBH from H in METERS = H(ft)/3.28084."""
    SUBSTITUTE_MAP = {60: 32, 63: 32}
    out = []
    for r in rows:
        sid_raw = int(r["SID"])
        sid = SUBSTITUTE_MAP.get(sid_raw, sid_raw)          # 60/63 -> 32 substitution
        pos = int(np.searchsorted(LE_CLASSES, sid))
        in_bundle = (pos < LE_CLASSES.size) and (LE_CLASSES[pos] == sid)
        if not in_bundle:                                   # MASK out-of-bundle (incl. 77)
            out.append(dict(SID=sid_raw, in_bundle=False, CR=np.nan, CBH=np.nan))
            continue
        enc = pos                                           # SID_ENC = searchsorted index (sorted)
        # baseline CR_pred from the bundle recipe (func4 for 'fit', constant for 'mean'), clipped
        kind, val = PARAMS_BY_SID[int(sid)]
        if kind == "fit":
            X = np.array([[r["DBH(inch)"]], [r["H(ft)"]]], dtype=float)   # [DBH, H] order
            cr_base = float(np.clip(func4_map(X, *np.asarray(val, float)), 0, 1)[0])
        else:
            cr_base = float(np.clip(val, 0, 1))
        feat = {
            "H(ft)": r["H(ft)"], "DBH(inch)": r["DBH(inch)"], "CD(%)": r["CD(%)"],
            "Elev(hm)": r["Elev(hm)"], "Slope(tan)": r["Slope(tan)"],
            "Azimuth(rad)": r["Azimuth(rad)"], "SID_ENC": enc,
            "Lat": r["Lat"], "Long": r["Long"], "CR_pred": cr_base,
        }
        X = pd.DataFrame([feat])[FEATURE_COLS].copy()
        X[NUMERIC_COLS] = SCALER.transform(X[NUMERIC_COLS])
        cr = float(XGB_MODEL.predict(X)[0])                 # predicted true crown ratio
        h_m = r["H(ft)"] / 3.28084                          # height in meters
        cbh = (1.0 - cr) * h_m                              # BUG FIX 1: CBH = H*(1-CR)
        out.append(dict(SID=sid_raw, sid_used=sid, in_bundle=True,
                        CR_base=round(cr_base, 4), CR=round(cr, 4),
                        H_m=round(h_m, 3), CBH=round(cbh, 3)))
    return pd.DataFrame(out)


# --- synthetic rows: several in-bundle SIDs (incl. a 'mean' SID 19 and substituted 60), and SID=77 ---
synthetic = [
    dict(SID=11, **dict(H_ft=14*3.28084, DBH_in=20*0.393701)),  # 소나무, fit
    dict(SID=32, **dict(H_ft=18*3.28084, DBH_in=24*0.393701)),  # 신갈나무, fit
    dict(SID=19, **dict(H_ft=10*3.28084, DBH_in=12*0.393701)),  # MEAN-recipe SID
    dict(SID=60, **dict(H_ft=16*3.28084, DBH_in=22*0.393701)),  # substituted 60->32
    dict(SID=68, **dict(H_ft=22*3.28084, DBH_in=30*0.393701)),  # fit
    dict(SID=77, **dict(H_ft=15*3.28084, DBH_in=21*0.393701)),  # mixed forest -> MASK
    dict(SID=82, **dict(H_ft=12*3.28084, DBH_in=18*0.393701)),  # non-target -> MASK
]
rows = []
for s in synthetic:
    rows.append(dict(
        SID=s["SID"], **{"H(ft)": s["H_ft"], "DBH(inch)": s["DBH_in"]},
        **{"CD(%)": 0.85, "Elev(hm)": 4.2, "Slope(tan)": 0.45,
           "Azimuth(rad)": 2.1, "Lat": 1_600_000.0, "Long": 950_000.0}))

res = predict_block(rows)
pd.set_option("display.width", 200)
print("\n=== prediction results ===")
print(res.to_string(index=False))

# ===== assertions =====
inb = res[res.in_bundle]
masked = res[~res.in_bundle]

assert (res["SID"] == 77).any(), "test must include SID=77"
m77 = res[res["SID"] == 77].iloc[0]
assert (not m77["in_bundle"]) and np.isnan(m77["CR"]) and np.isnan(m77["CBH"]), \
    "SID=77 must be masked (NaN CR/CBH)"
m82 = res[res["SID"] == 82].iloc[0]
assert (not m82["in_bundle"]) and np.isnan(m82["CR"]), "SID=82 must be masked"

assert len(inb) == 5, f"expected 5 in-bundle rows, got {len(inb)}"
for _, row in inb.iterrows():
    assert 0.0 <= row["CR"] <= 1.0, f"CR out of [0,1] for SID {row['SID']}: {row['CR']}"
    assert 0.0 <= row["CR_base"] <= 1.0, f"CR_base out of [0,1] for SID {row['SID']}"
    assert np.isfinite(row["CBH"]), f"CBH not finite for SID {row['SID']}"
    assert 0.0 <= row["CBH"] <= row["H_m"] + 1e-6, \
        f"CBH must be in [0,H] for SID {row['SID']}: CBH={row['CBH']} H={row['H_m']}"

# substitution check: SID=60 used sid 32
assert res[res["SID"] == 60].iloc[0]["sid_used"] == 32, "60 must be substituted to 32"

# cross-check: map-time func4 == training func4 for a fitted SID (bit-for-bit)
sid = 32
kind, val = PARAMS_BY_SID[sid]
assert kind == "fit"
Dn, Hn = 24 * 0.393701, 18 * 3.28084
ref = float(np.clip(func4_ref(np.array([[Dn], [Hn]]), *np.asarray(val, float)), 0, 1)[0])
mine = float(np.clip(func4_map(np.array([[Dn], [Hn]]), *np.asarray(val, float)), 0, 1)[0])
assert abs(ref - mine) < 1e-12, f"map func4 must match retrain func4: {ref} vs {mine}"
print(f"\nfunc4 cross-check (SID 32): retrain={ref:.10f}  map={mine:.10f}  delta={abs(ref-mine):.2e}")

print("\nALL ASSERTIONS PASSED")
