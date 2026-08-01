"""WP-B-4 — nationwide CBH/CR prediction from the PLOT-DISJOINT hybrid bundle.

Standalone extraction of the prediction path in src/CRModel3-Mapping.ipynb (cells: bundle load +
raster selection + windowed GPU prediction). The class-conditional Imsang SAMPLING cells are
deliberately NOT included — this script consumes the already-sampled F:\\CBH\\Clip_*_sampled.tif
rasters and must never regenerate/overwrite them.

Deploys the corrected model and the WP-B-3 fixes (see memory/decisions/2026-06-22_wp-b-readonly-gates.md):
  - plot-disjoint bundle (result/Baseline3/HM변형-try2.0-plotdisjoint/model/hybrid_plotdisjoint_SEED100.joblib)
  - CBH = H*(1-CR)            (was crown length)
  - Slope(tan) = tan(radians) (slope raster is radians; was tan(deg2rad(.)) → ~59x too small)
  - slope_file = Slope raster (was the DEM raster)
  - species_file = KOFTR_INT.tif; substitution 60/63→32; mask SIDs not in the 39-class bundle (incl. 77)

Outputs NEW tagged rasters under F:\\CBH — never the INVALID CBH4.tif/CR4.tif.
CRS EPSG:5179, 5 m grid. Run in geo-ml4 (CuPy/CUDA). No scipy curve_fit (params precomputed) → safe there.

Usage:
  # dry-run on a forested block range (writes *_DRYRUN_*.tif, easy to delete):
  python src/wp_b4_predict_maps.py --start 1820 --max-blocks 60 --out-suffix _DRYRUN
  # full nationwide extent:
  python src/wp_b4_predict_maps.py
"""
from __future__ import annotations
import os, argparse, traceback
from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from rasterio.transform import xy
import cupy as cp
from tqdm import tqdm
import joblib

# ----------------------------------------------------------------------------- config / provenance
RASTER_DIR = r"F:\CBH"                                          # nationwide rasters (EPSG:5179, 5 m)
BUNDLE_PATH = os.path.join(r"D:\ForestFire\CBH\result\Baseline3",
                           "HM변형-try2.0-plotdisjoint", "model",
                           "hybrid_plotdisjoint_SEED100.joblib")
OUT_TAG = "HM변형-try2.0-plotdisjoint_SEED100"                  # provenance tag for the corrected maps
SUBSTITUTE_MAP = {60: 32, 63: 32}                              # raster KOFTR_GROU code → in-bundle SID (decision 2026-06-22)
CM_TO_INCH = 0.393701                                          # DBH cm → inch (bundle training units)
M_TO_FT = 3.28084                                              # H m → ft
SEED = 100                                                     # bundle seed (no RNG at predict time; recorded for provenance)
STOP_ON_ERROR = True                                           # fail loud on the first block error (set False to skip-and-continue)


def write_log(log_content, block_num, initialize=False):       # append per-block progress to log_Prediction.txt
    mode = 'w' if (block_num == 0 or initialize) else 'a'      # truncate on first block / forced init
    stamp = datetime.now().strftime(r"%Y/%m/%d %H:%M:%S")      # human-readable timestamp
    with open(os.path.join(os.getcwd(), "log_Prediction.txt"), mode, encoding="utf-8") as f:
        f.write(f"{stamp}\t{log_content}\n")                   # one line per event


def create_valid_mask(block, nodata):                          # validity mask honoring the nodata convention
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        return ~cp.isnan(block)                                # NaN-nodata case
    return block != nodata                                     # sentinel-nodata case


def func4(X, a1, a2, a3, b):                                   # H&M(1996) logistic CR — EXACT copy of the retrain script
    H, D = X                                                   # NOTE: X column order is [DBH(inch), H(ft)] (mislabeled names, kept for parity)
    H_log, D_log = np.log1p(H), np.log1p(D)                    # log1p transforms
    z = (a1 * H_log / D_log) + (a2 * H_log) + (a3 * D_log ** 2) + b
    return 1.0 / (1.0 + np.exp(-z))                            # logistic → CR in (0,1)


def load_bundle():
    """Load the plot-disjoint bundle and build GPU lookup tables (mirrors mapping-notebook cell 31)."""
    bundle = joblib.load(BUNDLE_PATH)                          # dict, NOT a sklearn Pipeline
    xgb_model = bundle["xgb_model"]                            # XGBRegressor (10 features, FEATURE_COLS order)
    scaler = bundle["scaler"]                                  # StandardScaler fit on TRAIN NUMERIC_COLS
    params_by_sid = bundle["params_by_sid"]                    # {int(SID): ('fit', ndarray(4)) | ('mean', float)}
    feature_cols = list(bundle["FEATURE_COLS"])               # exact 10-col order
    numeric_cols = list(bundle["NUMERIC_COLS"])               # the 9 cols the scaler transforms
    le_classes = np.asarray(bundle["label_encoder_classes"], dtype=np.int64)  # 39 sorted SIDs
    assert (np.diff(le_classes) > 0).all(), "label_encoder_classes must be sorted for searchsorted"
    print(f"Loaded plot-disjoint bundle: {len(le_classes)} SIDs, SEED={bundle['SEED']}, mode={bundle['mode']}")

    # dense per-SID baseline recipe aligned to le_classes (drives CR_pred at map time)
    params = np.zeros((len(le_classes), 4), dtype=np.float64)  # func4 params per SID (valid where is_fit)
    is_fit = np.zeros(len(le_classes), dtype=bool)            # True → use func4, False → constant mean
    mean_cr = np.zeros(len(le_classes), dtype=np.float64)     # constant CR_pred for 'mean' SIDs
    for i, sid in enumerate(le_classes.tolist()):
        kind, val = params_by_sid[int(sid)]
        if kind == "fit":
            params[i] = np.asarray(val, dtype=np.float64); is_fit[i] = True
        else:
            mean_cr[i] = float(val)
    print(f"  baseline recipes: {int(is_fit.sum())} fit / {int((~is_fit).sum())} mean")

    gpu = dict(
        xgb_model=xgb_model, scaler=scaler, feature_cols=feature_cols, numeric_cols=numeric_cols,
        keys_sorted=cp.asarray(le_classes),                   # (39,) sorted int64 SIDs
        params_sorted=cp.asarray(params),                     # (39,4)
        is_fit=cp.asarray(is_fit),                            # (39,)
        mean_cr=cp.asarray(mean_cr),                          # (39,)
        sub_keys=cp.asarray(np.asarray(sorted(SUBSTITUTE_MAP), dtype=np.int64)),
        sub_vals=cp.asarray(np.asarray([SUBSTITUTE_MAP[k] for k in sorted(SUBSTITUTE_MAP)], dtype=np.int64)),
    )
    return gpu


def select_rasters():
    """Resolve the input raster paths by filename (mirrors cell 35, incl. the WP-B-3 slope/species fixes)."""
    rasters = glob(os.path.join(RASTER_DIR, "*.tif"))
    pick = lambda key: next((f for f in rasters if key in os.path.basename(f)), None)
    paths = dict(dem=pick("DEM"), slope=pick("Slope"), aspect=pick("Aspect"),
                 height=pick("HEIGHT"), dbh=pick("DMCLS_CD"), density=pick("DNST_CD"),
                 species=pick("KOFTR"))                        # KOFTR_INT.tif (int KOFTR_GROU codes)
    for name, p in paths.items():
        if p is None:
            raise FileNotFoundError(f"Raster of '{name}' not found in {RASTER_DIR}")
    assert paths["dem"] != paths["slope"], f"dem and slope must differ: {paths['dem']} vs {paths['slope']}"
    for k in ("dem", "slope", "species"):
        print(f"  {k:7s} = {os.path.basename(paths[k])}")
    return paths


def gen_tiles(width, height, win):
    """Tile the full grid into win×win windows (clipped at edges). Pixel results are window-independent
    (no cross-pixel ops), so this is identical to the native 128px blocks but with ~ (win/128)^2 fewer
    Python iterations — the dominant cost on a 663,518-block raster. Returns [(bid, Window), ...]."""
    out = []; bid = 0
    for r0 in range(0, height, win):
        h = min(win, height - r0)
        for c0 in range(0, width, win):
            out.append((bid, Window(c0, r0, min(win, width - c0), h))); bid += 1
    return out


def predict(gpu, paths, out_suffix="", start=0, max_blocks=0, win_size=0):
    """Windowed GPU prediction over the nationwide grid. Writes NEW tagged CR/CBH rasters."""
    out_names = {"CR": f"CR_{OUT_TAG}{out_suffix}.tif",
                 "CBH": f"CBH_meter_{OUT_TAG}{out_suffix}.tif"}  # CR 0-1, CBH meters
    # Safety: never write onto the invalid frozen products.
    for nm in out_names.values():
        assert nm not in ("CBH4.tif", "CR4.tif"), "refusing to overwrite frozen invalid maps"

    feature_cols, numeric_cols = gpu["feature_cols"], gpu["numeric_cols"]
    xgb_model, scaler = gpu["xgb_model"], gpu["scaler"]
    block_size = 512
    write_log("START PREDICTION", block_num=None, initialize=True)

    with rasterio.open(paths["species"]) as species_ras, \
         rasterio.open(paths["dem"]) as dem_ras, \
         rasterio.open(paths["slope"]) as slp_ras, \
         rasterio.open(paths["aspect"]) as asp_ras, \
         rasterio.open(paths["height"]) as h_ras, \
         rasterio.open(paths["dbh"]) as dbh_ras, \
         rasterio.open(paths["density"]) as cd_ras:

        ref_transform = dem_ras.transform
        profile = dem_ras.profile.copy()
        b_h = profile.get("blockysize", block_size); b_w = profile.get("blockxsize", block_size)
        dem_nodata = dem_ras.nodata if dem_ras.nodata is not None else -9999.
        h_nodata = h_ras.nodata if h_ras.nodata is not None else -9999.

        # all input rasters must share CRS=EPSG:5179, transform, and grid extent
        for src in (species_ras, slp_ras, asp_ras, h_ras, dbh_ras, cd_ras):
            assert src.crs == dem_ras.crs
            assert src.transform == dem_ras.transform
            assert (src.width, src.height) == (dem_ras.width, dem_ras.height)

        profile.update(driver="GTiff", dtype="float32", count=1, nodata=dem_nodata,
                       tiled=True, blockxsize=b_w, blockysize=b_h,
                       compress="ZSTD", predictor=3, bigtiff="YES")

        if win_size > 0:                                       # tile the grid into win_size² windows (fast path)
            all_blocks = gen_tiles(species_ras.width, species_ras.height, win_size)
        else:                                                  # native raster blocks (128px here) — slow but faithful
            all_blocks = [(i, win) for i, (ji, win) in enumerate(species_ras.block_windows(1))]
        total = len(all_blocks)
        sel = all_blocks[start:]                               # WP-B-4: full extent by default (start=0)
        if max_blocks:                                         # dry-run / partial: cap the count
            sel = sel[:max_blocks]
        print(f"blocks: total={total}  selected={len(sel)} (start={start}, max_blocks={max_blocks or 'all'}, "
              f"win_size={win_size or 'native'})  out={out_names}")

        dst_files = {attr: rasterio.open(os.path.join(RASTER_DIR, out_names[attr]), "w", **profile)
                     for attr in ("CR", "CBH")}
        valid_total = pred_total = 0                           # running coverage counters
        try:
            for bid, win in tqdm(sel, desc="Predict CBH/CR", total=len(sel)):
                try:
                    species_block = cp.asarray(species_ras.read(1, window=win))
                    dem_block = cp.asarray(dem_ras.read(1, window=win))
                    slp_block = cp.asarray(slp_ras.read(1, window=win))
                    asp_block = cp.asarray(asp_ras.read(1, window=win))
                    h_block = cp.asarray(h_ras.read(1, window=win))
                    dbh_block = cp.asarray(dbh_ras.read(1, window=win))
                    cd_block = cp.asarray(cd_ras.read(1, window=win))

                    out_dtype = cp.float32
                    out_cr = cp.full(dem_block.shape, dem_nodata, dtype=out_dtype)   # nodata until filled
                    out_cbh = cp.full(dem_block.shape, dem_nodata, dtype=out_dtype)

                    valid_mask = create_valid_mask(dem_block, dem_nodata) & create_valid_mask(h_block, h_nodata)
                    if not bool(valid_mask.any()):
                        dst_files["CR"].write(out_cr.get(), 1, window=win)
                        dst_files["CBH"].write(out_cbh.get(), 1, window=win)
                        continue

                    # unit conversions (valid pixels only) — match the bundle's training units
                    h_ft_block = cp.where(valid_mask, h_block * M_TO_FT, h_block)    # H m → ft
                    dbh_block = cp.where(valid_mask, dbh_block * CM_TO_INCH, dbh_block)  # DBH cm → inch
                    dem_block = cp.where(valid_mask, dem_block / 100.0, dem_block)   # Elev m → hectometer
                    slp_block = cp.where(valid_mask, cp.tan(slp_block), slp_block)   # WP-B-3: radians → tan(angle)
                    asp_block = cp.where(valid_mask, cp.deg2rad(asp_block), asp_block)  # Aspect deg → rad

                    # Lat/Long per pixel (center) on the shared affine grid
                    hh, ww = species_block.shape
                    rr, cc = np.meshgrid(np.arange(hh), np.arange(ww), indexing="ij")
                    aff = win_transform(win, ref_transform)
                    xs, ys = xy(aff, rr.ravel(), cc.ravel(), offset="center")
                    long_cp = cp.asarray(np.asarray(xs).reshape(hh, ww))
                    lat_cp = cp.asarray(np.asarray(ys).reshape(hh, ww))

                    # species code → substitution (60/63→32) → SID_ENC via searchsorted into sorted classes
                    sid_flat = species_block.ravel().astype(cp.int64, copy=False)
                    ps = cp.searchsorted(gpu["sub_keys"], sid_flat)
                    ms = (ps < gpu["sub_keys"].size) & (gpu["sub_keys"][cp.clip(ps, 0, gpu["sub_keys"].size - 1)] == sid_flat)
                    sid_mapped = sid_flat.copy(); sid_mapped[ms] = gpu["sub_vals"][ps[ms]]

                    v_flat = valid_mask.ravel()
                    pos = cp.searchsorted(gpu["keys_sorted"], sid_mapped)
                    posc = cp.clip(pos, 0, gpu["keys_sorted"].size - 1)
                    in_bundle = (pos < gpu["keys_sorted"].size) & (gpu["keys_sorted"][posc] == sid_mapped)
                    # COVERAGE: out-of-bundle SIDs (incl. 77 mixed forest ~16.84%, 78/81-94) stay nodata (masked)
                    pred_mask = v_flat & in_bundle

                    valid_total += int(valid_mask.sum().get())
                    block_pred = int(pred_mask.sum().get())
                    pred_total += block_pred
                    write_log(f"[block {bid}] valid={int(valid_mask.sum().get())} pred={block_pred}", bid)
                    if not bool(pred_mask.any()):
                        dst_files["CR"].write(out_cr.get(), 1, window=win)
                        dst_files["CBH"].write(out_cbh.get(), 1, window=win)
                        continue

                    idx = cp.nonzero(pred_mask)[0]              # predictable flat indices
                    enc_vec = pos[idx]                          # SID_ENC = sorted-class position
                    par = cp.take(gpu["params_sorted"], enc_vec, axis=0)   # (K,4)
                    isfit = cp.take(gpu["is_fit"], enc_vec)                 # (K,)
                    meancr = cp.take(gpu["mean_cr"], enc_vec)               # (K,)

                    h_ft_vec = h_ft_block.ravel()[idx]
                    h_m_vec = h_block.ravel()[idx]             # true height in METERS (for CBH)
                    dbh_vec = dbh_block.ravel()[idx]
                    cd_vec = cd_block.ravel()[idx]
                    dem_vec = dem_block.ravel()[idx]
                    slp_vec = slp_block.ravel()[idx]
                    asp_vec = asp_block.ravel()[idx]
                    lat_vec = lat_cp.ravel()[idx]
                    long_vec = long_cp.ravel()[idx]

                    # baseline CR_pred (func4 for 'fit' SIDs, constant for 'mean'); X = [DBH(inch), H(ft)]
                    Hc = cp.log1p(h_ft_vec); Dc = cp.log1p(dbh_vec)
                    Dc = cp.where(Dc == 0, cp.asarray(np.finfo(np.float64).eps), Dc)
                    a1, a2, a3, bb = par[:, 0], par[:, 1], par[:, 2], par[:, 3]
                    z = (a1 * Hc / Dc) + (a2 * Hc) + (a3 * Dc ** 2) + bb
                    cr_pred_fit = 1.0 / (1.0 + cp.exp(-z))
                    cr_pred_base = cp.clip(cp.where(isfit, cr_pred_fit, meancr), 0.0, 1.0)

                    df_input = pd.DataFrame({
                        "H(ft)": h_ft_vec.get(), "DBH(inch)": dbh_vec.get(), "CD(%)": cd_vec.get(),
                        "Elev(hm)": dem_vec.get(), "Slope(tan)": slp_vec.get(), "Azimuth(rad)": asp_vec.get(),
                        "SID_ENC": enc_vec.get().astype(np.int64), "Lat": lat_vec.get(), "Long": long_vec.get(),
                        "CR_pred": cr_pred_base.get(),
                    })
                    X = df_input[feature_cols].copy()
                    X[numeric_cols] = scaler.transform(X[numeric_cols])   # scale with TRAIN stats (SID_ENC as-is)
                    cr_pred = xgb_model.predict(X)             # predicted true crown ratio CR
                    cr_pred = np.clip(cr_pred, 0.0, 1.0)       # physical validity: XGB can over/undershoot;
                    #                                            CR∈[0,1] ⇒ CBH∈[0,H] (no CR>1 / negative CBH). WP-B-4.

                    out_cr.ravel()[idx] = cp.asarray(cr_pred, dtype=out_dtype)
                    cbh_pred = cp.asarray(1.0 - cr_pred) * h_m_vec         # WP-B-3 fix: CBH = H*(1-CR)
                    out_cbh.ravel()[idx] = cp.asarray(cbh_pred, dtype=out_dtype)

                    dst_files["CR"].write(out_cr.get(), 1, window=win)
                    dst_files["CBH"].write(out_cbh.get(), 1, window=win)

                except Exception as e:
                    write_log(f"BLOCK {bid} failed: {type(e).__name__}: {e}", bid)
                    traceback.print_exc()
                    if STOP_ON_ERROR:
                        raise
                    try:
                        dst_files["CR"].write(out_cr.get(), 1, window=win)
                        dst_files["CBH"].write(out_cbh.get(), 1, window=win)
                    except Exception:
                        pass
                    continue
        finally:
            for f in dst_files.values():
                f.close()

    pct = (100.0 * pred_total / valid_total) if valid_total else 0.0
    print(f"DONE. valid={valid_total:,} predicted={pred_total:,} ({pct:.1f}% of valid) masked={valid_total - pred_total:,}")
    print(f"wrote: {[os.path.join(RASTER_DIR, n) for n in out_names.values()]}")


def main():
    ap = argparse.ArgumentParser(description="WP-B-4 nationwide CBH/CR prediction (plot-disjoint bundle)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N block windows (0 = full extent)")
    ap.add_argument("--max-blocks", type=int, default=0, help="process at most N blocks (0 = all); use for a dry-run")
    ap.add_argument("--out-suffix", type=str, default="", help="suffix on output filenames (e.g. _DRYRUN)")
    ap.add_argument("--win-size", type=int, default=0,
                    help="tile the grid into N×N windows (0 = native 128px blocks; 1024 recommended for speed)")
    args = ap.parse_args()
    np.random.seed(SEED); cp.random.seed(SEED)                # determinism (no RNG at predict time; recorded anyway)
    gpu = load_bundle()
    paths = select_rasters()
    predict(gpu, paths, out_suffix=args.out_suffix, start=args.start,
            max_blocks=args.max_blocks, win_size=args.win_size)


if __name__ == "__main__":
    main()
