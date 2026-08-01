"""WP-0 Branch B — leakage-corrected re-train (plot-disjoint CV).

Reproduces the existing CBH/CR pipeline faithfully and changes ONLY the data partitioning, so the
leakage effect can be measured as old(tree-level) vs new(plot-level) metrics on identical model code.

- Baseline: Hasenauer & Monserud (1996) logistic crown-ratio model `func4`, fit per species via
  curve_fit + L2-regularized minimize (lam=0), exactly as src/CRModel3 / CRModel4-ResidualLearning.
- SPLIT_MODE = "tree": original behavior — per-cycle random row sample (validation target: must
  reproduce the published Evaluation_HM변형-try1.0.csv numbers).
- SPLIT_MODE = "plot": GroupShuffleSplit / StratifiedGroupKFold grouped by the verified plot key
  `SampleID`, with a mandatory train∩test plot-disjoint assertion (rules/cv-integrity.md).

Read-only on inputs; writes a NEW tagged result dir (never overwrites a frozen snapshot).
Run:  python src/wp0_plot_disjoint_retrain.py --mode tree   (then)   --mode plot
"""
from __future__ import annotations
import os, argparse, warnings                                   # CLI + path + warning control
import numpy as np                                              # arrays, math, RNG
import pandas as pd                                             # tabular I/O
from scipy.optimize import curve_fit, minimize                  # baseline fitting (as in notebooks)
from sklearn.metrics import r2_score, mean_absolute_error       # metrics
from sklearn.model_selection import (StratifiedKFold,           # tree-level CV (original)
                                     GroupShuffleSplit,         # plot-disjoint hold-out (new)
                                     StratifiedGroupKFold)      # plot-disjoint CV (new)

SEED = 100                                                      # matches CRModel3/CRModel4 (reproducibility)
TEST_RATIO = 0.3                                                # matches notebooks
N_FOLD = 5                                                      # matches notebooks
PARAMS_NUM = 4                                                  # func4 has 4 parameters
PLOT_KEY = "SampleID"                                           # verified NFI plot key (= 표본점번호)
DATA = "data/NFI6-7_cleaned2.csv"                               # cleaned model input (read-only)
# Reproducibility nudge: the published run used curve_fit's default p0=[1,1,1,1], which in THIS scipy
# build converges to a garbage basin for large species (params ~ -200 → saturated logistic → R²≈-11).
# Seeding p0 here lands in the SAME optimum as the published numbers (validated: SID 11 0.121 vs 0.121,
# SID 32 0.082 vs 0.082). It changes the starting point only, not the model. (WP-D: note this in README.)
P0 = [3.0, -1.0, 0.1, -1.0]                                     # initial guess for func4 (a1,a2,a3,b)


def func4(X, a1, a2, a3, b):                                    # Hasenauer & Monserud (1996) logistic CR
    H, D = X                                                    # NOTE: column order [DBH(inch), H(ft)] as in notebook
    H_log, D_log = np.log1p(H), np.log1p(D)                     # log1p transforms (as original)
    z = (a1 * H_log / D_log) + (a2 * H_log) + (a3 * D_log ** 2) + b  # linear predictor
    return 1.0 / (1.0 + np.exp(-z))                             # logistic squashing → CR in (0,1)


def loss_func4(params, lam, X, y):                              # regularized SSE used by minimize()
    y_pred = func4(X, *params)                                  # model prediction
    return np.sum((y - y_pred) ** 2) + lam * np.sum(np.asarray(params) ** 2)  # SSE + L2 (lam=0 → plain SSE)


def load():                                                     # load cleaned data, keep needed columns
    df = None                                                   # placeholder until a read succeeds
    for enc in ("cp949", "utf-8"):                              # cleaned CSV is cp949-encoded
        try:
            df = pd.read_csv(DATA, encoding=enc); break          # stop at first successful decode
        except Exception:
            continue                                            # try the next encoding
    if df is None:                                              # neither encoding worked
        raise SystemExit(f"cannot read {DATA}")                 # fail loudly (no fabrication)
    need = ["SID", PLOT_KEY, "Cycle", "DBH(inch)", "H(ft)", "CR"]  # minimal columns for the baseline
    df = df.dropna(subset=need)                                 # drop rows missing any essential field
    return df[need + [c for c in df.columns if c not in need]]  # essentials first, keep the rest


def split_species(dfx, mode, rng):                              # build train/test for ONE species
    """Return (train_df, test_df). 'tree' = per-cycle random rows; 'plot' = disjoint plots."""
    if mode == "tree":                                          # ORIGINAL leaky behavior
        tr_parts, te_parts = [], []                             # accumulate per-cycle pieces
        for cyc, g in dfx.groupby("Cycle"):                     # split within each NFI cycle (as notebook)
            te = g.sample(frac=TEST_RATIO, random_state=SEED)   # random ROWS → trees, not plots
            tr_parts.append(g.drop(te.index)); te_parts.append(te)  # remainder is train
        return pd.concat(tr_parts), pd.concat(te_parts)         # combined train / test
    # mode == "plot": hold out whole plots, never splitting a plot across train/test
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)  # group-aware hold-out
    tr_idx, te_idx = next(gss.split(dfx, groups=dfx[PLOT_KEY]))  # indices by plot membership
    tr, te = dfx.iloc[tr_idx], dfx.iloc[te_idx]                 # materialize the two frames
    assert set(tr[PLOT_KEY]).isdisjoint(set(te[PLOT_KEY]))      # MANDATORY: no plot in both (cv-integrity.md)
    return tr, te


def fit_baseline(train, mode):                                  # fit func4 on a species' training data
    Xtr = train[["DBH(inch)", "H(ft)"]].values.T               # shape (2, n): row0=DBH, row1=H (as notebook)
    ytr = train["CR"].values                                   # target crown ratio
    cyc = train["Cycle"].values                                # stratification label (NFI cycle)
    grp = train[PLOT_KEY].values                               # plot groups (used only in plot mode)
    if mode == "tree":                                         # original: row-wise StratifiedKFold by cycle
        splitter = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
        split_args = (Xtr.T, cyc)                              # stratify by cycle, no grouping (leaky)
    else:                                                       # new: plot-grouped CV, stratified by cycle
        splitter = StratifiedGroupKFold(n_splits=N_FOLD)        # no plot crosses a fold boundary
        split_args = (Xtr.T, cyc, grp)                         # stratify by cycle, group by SampleID
    try:                                                       # build folds; tiny/imbalanced species can fail
        folds = list(splitter.split(*split_args))              # materialize fold indices
    except ValueError:                                         # not enough members/groups per class for N_FOLD
        folds = [(np.arange(len(ytr)), np.array([], dtype=int))]  # fall back to a single all-data fit
    best_score, best_params = -np.inf, None                    # track best fold (by its train fit, as original)
    for tr_i, _ in folds:                                      # original scores on the fold's own train set
        try:
            popt, _ = curve_fit(func4, Xtr[:, tr_i], ytr[tr_i], p0=P0, maxfev=10000)  # NLS from seeded p0
            res = minimize(loss_func4, x0=popt, args=(0, Xtr[:, tr_i], ytr[tr_i]))  # refine (lam=0)
            score = r2_score(ytr[tr_i], func4(Xtr[:, tr_i], *res.x))  # train-fold R² (original's selection rule)
        except Exception:
            continue                                           # a degenerate fold is skipped, not faked
        if score > best_score:                                 # keep the best-fitting fold's parameters
            best_score, best_params = score, res.x
    return best_params                                         # may be None if every fold failed


def _metrics(y, yhat):                                          # shared metric set (honest: R²+RMSE+MAE+bias)
    e = yhat - y                                                # signed error (pred − obs)
    return dict(n=int(len(y)), R2=float(r2_score(y, yhat)),     # raw R² (negatives kept, see R2-9)
                RMSE=float(np.sqrt(np.mean(e ** 2))),           # root mean squared error
                MAE=float(np.mean(np.abs(e))),                  # mean absolute error
                bias=float(np.mean(e)))                         # mean signed error (over/under-prediction)


def evaluate(params, frame):                                    # baseline metrics on a frame
    if params is None or len(frame) == 0:                      # nothing to score
        return dict(n=int(len(frame)), R2=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan)  # honest NaN
    X = frame[["DBH(inch)", "H(ft)"]].values.T                 # same feature order as fit
    return _metrics(frame["CR"].values, func4(X, *params))     # raw (unclipped) baseline metrics


FEATURE_COLS = ["H(ft)", "DBH(inch)", "CD(%)", "Elev(hm)", "Slope(tan)",  # exact hybrid feature set
                "Azimuth(rad)", "SID_ENC", "Lat", "Long", "CR_pred"]      # (from CRModel3), incl. CR_pred
NUMERIC_COLS = [c for c in FEATURE_COLS if c != "SID_ENC"]      # everything except the encoded species id


def global_split(df, mode):
    """Return boolean test-mask for a GLOBAL split (all species together), per the hybrid's design.

    tree  = stratified random rows (original: train_test_split test_size=0.2, stratify=species).
    plot  = GroupShuffleSplit(test_size=0.2) on SampleID, with the disjoint assertion.
    """
    from sklearn.model_selection import train_test_split        # imported locally (tree path only)
    idx = np.arange(len(df))                                    # positional indices
    if mode == "tree":                                          # ORIGINAL leaky split
        tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=df["SID"].values)
    else:                                                       # plot-disjoint split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(idx, groups=df[PLOT_KEY].values))
        assert set(df[PLOT_KEY].values[tr]).isdisjoint(set(df[PLOT_KEY].values[te]))  # MANDATORY
    mask = np.zeros(len(df), dtype=bool); mask[te] = True       # True where row is in the test set
    return mask


def make_cr_pred(df, test_mask, mode):
    """Baseline CR_pred for every row, fit per species on TRAIN ONLY (so test CR_pred is leakage-free).

    Falls back to the species' train-mean CR when a species cannot be fit (too few train plots).
    Returns (cr_pred, params_by_sid). params_by_sid maps {int(SID): ('fit', np.ndarray) |
    ('mean', float)} — the exact train-only baseline recipe per species, so CR_pred can be
    regenerated at map time WITHOUT re-fitting (and identically to this run).
    """
    cr_pred = np.full(len(df), np.nan)                          # output column, NaN until filled
    params_by_sid = {}                                          # per-species baseline recipe (map-time reuse)
    tr_global = ~test_mask                                      # training rows (global)
    for sid, idx in df.groupby("SID").indices.items():          # per species (positional indices)
        rows = np.asarray(idx)                                  # this species' row positions
        tr_rows = rows[tr_global[rows]]                         # its training rows
        if len(tr_rows) == 0:                                   # species absent from train
            continue                                            # leave NaN → dropped later
        train_sp = df.iloc[tr_rows]                             # this species' training frame
        # Fit only if enough train rows AND enough plots/cycles for N_FOLD CV; else fall back to mean.
        enough = (len(tr_rows) >= N_FOLD and
                  train_sp[PLOT_KEY].nunique() >= N_FOLD and
                  train_sp["Cycle"].nunique() >= 2)             # StratifiedKFold needs ≥2 classes
        params = fit_baseline(train_sp, mode) if enough else None  # robust: skip CV for tiny species
        if params is None:                                      # too small or degenerate fit
            mean_cr = float(train_sp["CR"].mean())             # fallback: species train-mean CR
            cr_pred[rows] = mean_cr                             # apply to all rows of the species
            params_by_sid[int(sid)] = ("mean", mean_cr)        # record the fallback recipe
        else:
            Xall = df.iloc[rows][["DBH(inch)", "H(ft)"]].values.T  # predict for ALL rows of the species
            cr_pred[rows] = np.clip(func4(Xall, *params), 0, 1)  # baseline prediction (clipped as original)
            params_by_sid[int(sid)] = ("fit", np.asarray(params, dtype=float))  # record fitted func4 params
    return cr_pred, params_by_sid


def run_hybrid(df, mode, save_model=False):
    """Global XGBoost hybrid; returns (per-species DataFrame, overfitting summary dict).

    When save_model=True AND mode=="plot", persists a self-contained joblib bundle (the trained
    XGB model, the train-fit scaler, per-species baseline recipes, the species label-encoder
    classes, and the feature schema) so WP-B can deploy the CORRECTED model to the nationwide maps.
    Only the plot-disjoint model is ever persisted — the leaky tree-mode model must not be shipped.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder  # local imports (hybrid path only)
    from xgboost import XGBRegressor                            # matches notebook (xgboost 2.1.4)

    df = df.reset_index(drop=True).copy()                       # clean positional index
    test_mask = global_split(df, mode)                          # one split for all species
    le = LabelEncoder()                                         # keep a reference so its classes can be saved
    df["SID_ENC"] = le.fit_transform(df["SID"])                # encode species id (not leakage)
    df["CR_pred"], params_by_sid = make_cr_pred(df, test_mask, mode)  # leakage-free baseline feature + recipes
    df = df.dropna(subset=FEATURE_COLS + ["CR"]).reset_index(drop=True)  # drop rows w/o a CR_pred etc.
    # Recompute the split on the filtered frame so masks stay aligned after dropna (same SEED → deterministic).
    test_mask = global_split(df, mode)                          # boolean test mask on the filtered df
    tr, te = ~test_mask, test_mask                              # train / test boolean masks

    scaler = StandardScaler()                                   # scale numeric features (fit on TRAIN only)
    Xtr = df.loc[tr, FEATURE_COLS].copy(); Xte = df.loc[te, FEATURE_COLS].copy()
    Xtr[NUMERIC_COLS] = scaler.fit_transform(Xtr[NUMERIC_COLS])  # fit+transform train
    Xte[NUMERIC_COLS] = scaler.transform(Xte[NUMERIC_COLS])     # transform test with train stats
    ytr, yte = df.loc[tr, "CR"].values, df.loc[te, "CR"].values  # targets

    model = XGBRegressor(random_state=SEED, eval_metric="rmse")  # defaults, matches notebook
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)  # capture train vs test curve
    ev = model.evals_result_                                    # {validation_0: train, validation_1: test}
    yhat = model.predict(Xte)                                   # held-out predictions

    if save_model and mode == "plot":                           # ship ONLY the corrected (plot-disjoint) model
        import joblib                                           # available in geo-ml3
        bundle = dict(xgb_model=model,                          # trained XGB residual/CR model
                      scaler=scaler,                            # StandardScaler fit on TRAIN numeric cols
                      params_by_sid=params_by_sid,              # per-species baseline recipe (CR_pred at map time)
                      FEATURE_COLS=FEATURE_COLS,                # exact feature order the model expects
                      NUMERIC_COLS=NUMERIC_COLS,                # which features the scaler transforms
                      label_encoder_classes=list(le.classes_),  # SID → SID_ENC mapping (sorted classes)
                      P0=list(P0), SEED=SEED, mode=mode)        # reproducibility provenance
        mdir = os.path.join("result", "Baseline3", "HM변형-try2.0-plotdisjoint", "model")  # new artifact dir
        os.makedirs(mdir, exist_ok=True)                        # create model/ subdir if missing
        bpath = os.path.join(mdir, f"hybrid_plotdisjoint_SEED{SEED}.joblib")  # NEW file (not a frozen overwrite)
        joblib.dump(bundle, bpath)                              # persist the deployable bundle
        print(f"  saved plot-disjoint hybrid bundle -> {bpath}  "
              f"(species recipes={len(params_by_sid)}, features={len(FEATURE_COLS)})")

    sid_te = df.loc[te, "SID"].values                           # per-species breakdown on the test set
    rows = []                                                   # per-species metric records
    for sid in np.unique(sid_te):                               # one row per species present in test
        m = sid_te == sid                                       # this species' test rows
        if m.sum() < 5:                                         # too few to score meaningfully
            continue
        mm = _metrics(yte[m], yhat[m])                          # full metric set for this species
        rows.append(dict(SID=int(sid), mode=mode, model="hybrid_xgb",
                         n_test=int(m.sum()),
                         R2_Test=round(mm["R2"], 4), RMSE_Test=round(mm["RMSE"], 4),
                         MAE_Test=round(mm["MAE"], 4), Bias_Test=round(mm["bias"], 4)))
    per_sp = pd.DataFrame(rows).sort_values("SID")              # per-species table
    summary = dict(mode=mode,                                   # overfitting evidence (R1-2)
                   R2_Test_pooled=round(float(r2_score(yte, yhat)), 4),
                   n_train=int(tr.sum()), n_test=int(te.sum()),
                   train_rmse_final=round(float(ev["validation_0"]["rmse"][-1]), 5),
                   test_rmse_final=round(float(ev["validation_1"]["rmse"][-1]), 5))
    summary["overfit_gap"] = round(summary["test_rmse_final"] - summary["train_rmse_final"], 5)
    return per_sp, summary


def main():
    ap = argparse.ArgumentParser()                              # parse the split mode + model
    ap.add_argument("--mode", choices=["tree", "plot"], required=True)  # tree=old(leaky), plot=corrected
    ap.add_argument("--model", choices=["baseline", "hybrid"], default="baseline")  # which model to run
    ap.add_argument("--min-n", type=int, default=30)            # skip species with too few trees
    ap.add_argument("--save-model", action="store_true",       # persist deployable plot-disjoint bundle
                    help="persist the plot-disjoint hybrid bundle for WP-B (hybrid + plot mode only)")
    args = ap.parse_args()

    if args.model == "hybrid":                                  # ---- Stage 2: XGB hybrid ----
        warnings.filterwarnings("ignore"); np.random.seed(SEED)  # determinism
        df = load()                                             # cleaned data
        per_sp, summary = run_hybrid(df, args.mode, save_model=args.save_model)  # run the global hybrid
        tag = "HM변형-try2.0-plotdisjoint"                      # NEW immutable tag
        odir = os.path.join("result", "Baseline3", tag); os.makedirs(odir, exist_ok=True)
        path = os.path.join(odir, f"hybrid_eval_{args.mode}_SEED{SEED}.csv")
        per_sp.to_csv(path, index=False, encoding="utf-8-sig")  # per-species table
        print(f"[hybrid {args.mode}] species={len(per_sp)}  pooled R2_Test={summary['R2_Test_pooled']}  "
              f"median per-sp R2_Test={per_sp.R2_Test.median():.4f}  R2<=0: {(per_sp.R2_Test<=0).sum()}/{len(per_sp)}")
        print(f"  overfit: train_rmse={summary['train_rmse_final']} test_rmse={summary['test_rmse_final']} "
              f"gap={summary['overfit_gap']}  (n_train={summary['n_train']} n_test={summary['n_test']})")
        print("wrote", path)
        return

    warnings.filterwarnings("ignore")                           # silence curve_fit/overflow noise
    np.random.seed(SEED)                                        # global determinism (recorded in output)
    rng = np.random.default_rng(SEED)                           # local RNG (unused branch reserved)
    df = load()                                                 # read cleaned data once

    rows = []                                                   # per-species result records
    for sid, dfx in df.groupby("SID"):                          # one model per species id
        if len(dfx) < args.min_n or dfx[PLOT_KEY].nunique() < N_FOLD:  # need enough trees AND plots
            continue                                            # skip low-n species (as original)
        train, test = split_species(dfx, args.mode, rng)        # build the (mode-specific) partition
        params = fit_baseline(train, args.mode)                 # fit the HM baseline on training only
        m_tr, m_te = evaluate(params, train), evaluate(params, test)  # train + held-out metrics
        rows.append(dict(SID=int(sid), mode=args.mode,          # record everything needed for the delta
                         n_train=m_tr["n"], n_test=m_te["n"],   # integer counts (R2-10)
                         plots_train=int(train[PLOT_KEY].nunique()),
                         plots_test=int(test[PLOT_KEY].nunique()),
                         R2_Train=round(m_tr["R2"], 4), R2_Test=round(m_te["R2"], 4),
                         RMSE_Test=round(m_te["RMSE"], 4), MAE_Test=round(m_te["MAE"], 4),
                         Bias_Test=round(m_te["bias"], 4)))

    out = pd.DataFrame(rows).sort_values("SID")                 # assemble the table
    tag = "HM변형-try2.0-plotdisjoint"                          # NEW run tag (immutable; never overwrite)
    odir = os.path.join("result", "Baseline3", tag)            # tagged output dir
    os.makedirs(odir, exist_ok=True)                            # create if missing
    path = os.path.join(odir, f"baseline_eval_{args.mode}_SEED{SEED}.csv")  # mode in filename
    out.to_csv(path, index=False, encoding="utf-8-sig")        # write the per-species table
    # Console summary: pooled medians (R² distribution is skewed; median is the honest center).
    print(f"[{args.mode}] species={len(out)}  "
          f"median R2_Test={out.R2_Test.median():.4f}  "
          f"mean R2_Test={out.R2_Test.mean():.4f}  "
          f"R2_Test<=0: {(out.R2_Test <= 0).sum()}/{len(out)}")
    print("wrote", path)


if __name__ == "__main__":
    main()
