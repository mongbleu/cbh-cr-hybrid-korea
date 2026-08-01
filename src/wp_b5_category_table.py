"""WP-B-5 — category (SID×AGECLS×DMCLS) CBH table from the PLOT-DISJOINT nationwide map.

Standalone extraction of the `analyze_raster_categorywise` path in src/Evaluation.ipynb (the cell that
writes `Pred-NIFS-ComparebyCategory_CR4.csv`). Rebuilds that table from the NEW, corrected CBH map so the
box-plots in `paper-Visualize.ipynb` reflect the plot-disjoint model instead of the INVALID CBH4.tif.

What changes vs. the notebook (see memory/sessions/2026-06-23_session-handoff.md):
  - pred raster: F:\\CBH\\CBH4.tif (INVALID)  →  F:\\CBH\\CBH_meter_hybrid_model.tif
  - output:      Pred-NIFS-ComparebyCategory_CR4.csv  →  pred_vs_NIFS_by_category.csv
                 (NEW tagged file under the plot-disjoint result dir; the OLD CR4 CSV is left untouched)

IMPORTANT data-flow note (verified 2026-06-23):
  The box-plot cell reads columns [target_feat, 'mean', 'NIFS-CBH']. This script REGENERATES the
  prediction side (mean/median/Q2/Q3/min/max per category) from the new map. The 'NIFS-CBH' column is a
  STATIC reference value per (SID,AGECLS,DMCLS) — it is NOT raster-derived (it was hand-entered in the
  old CR4 CSV) and is therefore CARRIED FORWARD UNCHANGED by a left-join on [SID,AGECLS,DMCLS]. The model
  correction changes the prediction, not the NIFS reference, so the comparison stays apples-to-apples.

This aggregation is pure numpy + rasterio block reads (NO CuPy, NO scipy curve_fit), so it runs in either
geo-ml3 or geo-ml4 WITHOUT a GPU. It is still a full-raster scan over 4 rasters (new CBH + KOFTR_INT +
AGCLS_INT + DMCLS_INT, EPSG:5179, 5 m) → expect tens of minutes of I/O. Confirm before launching the full
run (same gating as WP-B-4); use --dry-run first.

Usage:
  python src/wp_b5_category_table.py --dry-run        # cap to a few hundred blocks, writes *_DRYRUN.csv
  python src/wp_b5_category_table.py                  # full nationwide scan → tagged CSV
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from collections import defaultdict
from tqdm import tqdm

# ----------------------------------------------------------------------------- config / provenance
RASTER_DIR = r"F:\CBH"
PRED_MAP   = os.path.join(RASTER_DIR, "CBH_meter_hybrid_model.tif")  # NEW corrected map
SPECIES    = os.path.join(RASTER_DIR, "KOFTR_INT.tif")    # SID (KOFTR group) codes
AGE_PATH   = os.path.join(RASTER_DIR, "AGCLS_INT.tif")    # age class
DBH_PATH   = os.path.join(RASTER_DIR, "DMCLS_INT.tif")    # DBH class

OLD_CR4_CSV = os.path.join(r"D:\ForestFire\CBH\result\Baseline3",
                           "Pred-NIFS-ComparebyCategory_CR4.csv")          # source of the static NIFS-CBH ref
OUT_DIR     = os.path.join(r"D:\ForestFire\CBH\result\Baseline3", "hybrid-model")
OUT_TAG     = "hybrid_model"
OUT_NAME    = "pred_vs_NIFS_by_category.csv"

# Category grid — EXACT copy from Evaluation.ipynb `analyze_raster_categorywise` (parity).
SPECIES_LIST = [11, 12, 14, 15, 10, 31, 33, 32, 30, 77]
AGE_DBH_LIST = [[2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [4, 3], [5, 1],
                [5, 2], [5, 3], [6, 1], [6, 2], [6, 3], [7, 1], [7, 2],
                [7, 3], [8, 2], [8, 3], [9, 2], [9, 3]]
BINS, VMIN, VMAX = 20, 0, 30                                # notebook's category-table call used bins=20


def gen_tiles(width, height, win):
    """Tile the grid into win×win windows (clipped at edges). Aggregation is pixel-wise with NO cross-pixel
    ops, so tiled windows are identical to native 128px blocks but with far fewer Python iterations
    (same equivalence wp_b4 verified). Returns [Window, ...]."""
    out = []
    for r0 in range(0, height, win):
        h = min(win, height - r0)
        for c0 in range(0, width, win):
            out.append(Window(c0, r0, min(win, width - c0), h))
    return out


def analyze_raster_categorywise(pred_path, species_path, age_path, dbh_path,
                                bins=BINS, vmin=VMIN, vmax=VMAX, max_blocks=0, win_size=0):
    """Block-wise histogram accumulation per (SID,AGECLS,DMCLS); mean/median/Q2/Q3/min/max via histogram.
    Verbatim port of the notebook function (CPU/numpy), plus an optional --dry-run block cap and a
    win_size fast path (tile into win_size² windows instead of native 128px blocks)."""
    spe_age_dbh = [[i] + j for i in SPECIES_LIST for j in AGE_DBH_LIST]

    with rasterio.open(pred_path) as rpred, rasterio.open(species_path) as rsid, \
         rasterio.open(age_path) as rage, rasterio.open(dbh_path) as rdbh:
        prof = rpred.profile
        b_h = prof.get('blockysize', 512); b_w = prof.get('blockxsize', 512)
        height, width = rpred.height, rpred.width
        block_cnt = int(np.ceil(height / b_h)) * int(np.ceil(width / b_w))
        pred_nodata, sid_nodata = rpred.nodata, rsid.nodata
        age_nodata, dbh_nodata = rage.nodata, rdbh.nodata
        assert rpred.shape == rsid.shape == rage.shape == rdbh.shape, "all rasters must share the grid"

        edges = np.linspace(vmin, vmax, bins + 1)
        acc = defaultdict(lambda: {"n": 0, "sum_pred": 0.0, "min": 0.0, "max": 0.0,
                                   "hist_pred": np.zeros(bins, dtype=np.int64)})

        if win_size > 0:                                   # fast path: tile into win_size² windows
            windows = gen_tiles(width, height, win_size)
        else:                                              # native raster blocks (128px) — slow but faithful
            windows = [w for _, w in rpred.block_windows(1)]
        if max_blocks:                                     # dry-run: cap from the forested middle of the grid
            mid = len(windows) // 2
            windows = windows[mid:mid + max_blocks]
        for window in tqdm(windows, desc="Categorywise scan", total=len(windows)):
            pred = rpred.read(1, window=window, masked=True).astype(float)
            sid = rsid.read(1, window=window, masked=True)
            age = rage.read(1, window=window, masked=True)
            dbh = rdbh.read(1, window=window, masked=True)

            valid = ~pred.mask & np.isfinite(pred)
            if pred_nodata is not None and np.isfinite(pred_nodata):
                valid &= (pred != pred_nodata)
            valid &= ~sid.mask
            if sid_nodata is not None:
                valid &= (sid != sid_nodata)
            valid &= ~age.mask
            if age_nodata is not None:
                valid &= (age != age_nodata)
            if not valid.any():
                continue

            for cond in spe_age_dbh:
                mask = valid & (sid.astype(int) == cond[0]) & (age.astype(int) == cond[1]) & (dbh.astype(int) == cond[2])
                if not mask.any():
                    continue
                pm = pred[mask]
                key = f"{cond[0]}-{cond[1]}-{cond[2]}"
                a = acc[key]
                a["n"] += pm.size
                a["sum_pred"] += pm.sum()
                a["min"] = np.minimum(a["min"], np.min(pm))
                a["max"] = np.maximum(a["max"], np.max(pm))
                idx = np.searchsorted(edges, np.clip(pm, vmin, vmax), side="right") - 1
                np.add.at(a["hist_pred"], np.clip(idx, 0, bins - 1), 1)

    # ----- final per-category representative values (histogram quantiles) -----
    out = {}
    for key, a in acc.items():
        if a["n"] == 0:
            continue
        mean_pred = a["sum_pred"] / a["n"]
        c = a["hist_pred"].cumsum()

        def q(p):
            k = np.searchsorted(c, p * c[-1])
            return 0.5 * (edges[k] + edges[min(k + 1, bins)])

        out[key] = dict(mean=mean_pred, median=q(0.5), Q2=q(0.25), Q3=q(0.75),
                        minimum=a["min"], maximum=a["max"])
    return out


def build_table(cat, dry=False):
    """Assemble the category dataframe and left-join the STATIC NIFS-CBH reference from the old CR4 CSV."""
    df_cat = pd.DataFrame(cat).T.reset_index()
    new_data = np.array([np.asarray(x.split('-')) for x in df_cat['index'].values]).astype('int')
    df_new = pd.concat([pd.DataFrame(columns=['SID', 'AGECLS', 'DMCLS'], data=new_data), df_cat], axis=1)
    df_new = df_new.sort_values(by=['SID', 'AGECLS', 'DMCLS']).reset_index(drop=True)

    # carry forward the static NIFS reference (+ Korean Species label) from the old hand-built CR4 CSV
    if os.path.exists(OLD_CR4_CSV):
        old = pd.read_csv(OLD_CR4_CSV, encoding='cp949')
        keep = ['SID', 'AGECLS', 'DMCLS', 'NIFS-CBH']
        if 'Species' in old.columns:
            keep.insert(3, 'Species')
        old = old[[c for c in keep if c in old.columns]].copy()
        old = old.dropna(subset=['SID', 'AGECLS', 'DMCLS'])    # old CSV has blank hand-built trailing rows
        for k in ('SID', 'AGECLS', 'DMCLS'):
            old[k] = old[k].astype(int)
        df_new = df_new.merge(old, on=['SID', 'AGECLS', 'DMCLS'], how='left')
        matched = df_new['NIFS-CBH'].notna().sum()
        print(f"NIFS-CBH carried forward for {matched}/{len(df_new)} categories (left-join on SID-AGECLS-DMCLS)")
    else:
        print(f"WARNING: old CR4 CSV not found at {OLD_CR4_CSV} — NIFS-CBH column will be absent")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = OUT_NAME.replace(".csv", "_DRYRUN.csv") if dry else OUT_NAME
    out_path = os.path.join(OUT_DIR, out_name)
    assert os.path.basename(out_path) != "Pred-NIFS-ComparebyCategory_CR4.csv", "refusing to overwrite the old CSV"
    df_new.to_csv(out_path, index=False, encoding='cp949')        # cp949 to match the box-plot reader
    print(f"wrote {out_path}  ({len(df_new)} categories)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="WP-B-5 category CBH table from the plot-disjoint map")
    ap.add_argument("--dry-run", action="store_true", help="cap to ~300 mid-grid windows; write *_DRYRUN.csv")
    ap.add_argument("--win-size", type=int, default=1024,
                    help="tile into N×N windows (0 = native 128px blocks; 1024 default for speed)")
    args = ap.parse_args()
    for p in (PRED_MAP, SPECIES, AGE_PATH, DBH_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    print(f"pred map: {os.path.basename(PRED_MAP)}  (win_size={args.win_size or 'native'})")
    cat = analyze_raster_categorywise(PRED_MAP, SPECIES, AGE_PATH, DBH_PATH,
                                      max_blocks=300 if args.dry_run else 0, win_size=args.win_size)
    build_table(cat, dry=args.dry_run)


if __name__ == "__main__":
    main()
