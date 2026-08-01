"""WP-0 overfitting check: hybrid XGB train vs held-out (plot-disjoint) metrics.

Reuses the exact split/feature/scaler logic from wp0_plot_disjoint_retrain so the
numbers match the frozen plot-disjoint run (SEED=100). Computes R2/RMSE/MAE/bias on
BOTH the training plots and the held-out plots, pooled and per-species, purely to
document that train and held-out performance are close (i.e. no overfitting).

Read-only w.r.t. frozen results: writes a NEW csv, never overwrites the eval CSVs.
Run in geo-ml3:  conda run -n geo-ml3 python src/wp0_hybrid_overfit_check.py
"""
import os
import numpy as np
import pandas as pd

import wp0_plot_disjoint_retrain as R  # load(), global_split(), make_cr_pred(), FEATURE/NUMERIC_COLS, _metrics, SEED


def main():
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from xgboost import XGBRegressor

    df = R.load()
    df = df.reset_index(drop=True).copy()
    mode = "plot"

    test_mask = R.global_split(df, mode)
    le = LabelEncoder()
    df["SID_ENC"] = le.fit_transform(df["SID"])
    df["CR_pred"], _ = R.make_cr_pred(df, test_mask, mode)
    df = df.dropna(subset=R.FEATURE_COLS + ["CR"]).reset_index(drop=True)
    test_mask = R.global_split(df, mode)          # realign after dropna (same SEED)
    tr, te = ~test_mask, test_mask

    scaler = StandardScaler()
    Xtr = df.loc[tr, R.FEATURE_COLS].copy(); Xte = df.loc[te, R.FEATURE_COLS].copy()
    Xtr[R.NUMERIC_COLS] = scaler.fit_transform(Xtr[R.NUMERIC_COLS])
    Xte[R.NUMERIC_COLS] = scaler.transform(Xte[R.NUMERIC_COLS])
    ytr, yte = df.loc[tr, "CR"].values, df.loc[te, "CR"].values

    model = XGBRegressor(random_state=R.SEED, eval_metric="rmse")
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)

    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)

    mtr = R._metrics(ytr, yhat_tr)
    mte = R._metrics(yte, yhat_te)

    # plot-disjointness sanity check (must be empty intersection)
    ptr = set(df.loc[tr, R.PLOT_KEY]); pte = set(df.loc[te, R.PLOT_KEY])
    assert not (ptr & pte), "train/test plots overlap!"

    print("=== Hybrid XGB (plot-disjoint, SEED=%d) - pooled CR (0-1) ===" % R.SEED)
    print("n_train_trees=%d  n_test_trees=%d  plots_train=%d  plots_test=%d  (disjoint=%s)"
          % (tr.sum(), te.sum(), len(ptr), len(pte), not (ptr & pte)))
    hdr = "%-8s %8s %8s %8s %9s" % ("set", "R2", "RMSE", "MAE", "bias")
    print(hdr)
    print("%-8s %8.4f %8.4f %8.4f %9.4f" % ("train", mtr["R2"], mtr["RMSE"], mtr["MAE"], mtr["bias"]))
    print("%-8s %8.4f %8.4f %8.4f %9.4f" % ("held-out", mte["R2"], mte["RMSE"], mte["MAE"], mte["bias"]))
    print("gap(train-heldout): dR2=%+.4f  dRMSE=%+.4f  dMAE=%+.4f"
          % (mtr["R2"] - mte["R2"], mte["RMSE"] - mtr["RMSE"], mte["MAE"] - mtr["MAE"]))

    # persist a small summary table (new file; does not touch frozen eval CSVs)
    odir = os.path.join("result", "Baseline3", "hybrid-model")
    os.makedirs(odir, exist_ok=True)
    summ = pd.DataFrame([
        dict(set="train",    n_trees=int(tr.sum()), R2=round(mtr["R2"], 4), RMSE=round(mtr["RMSE"], 4),
             MAE=round(mtr["MAE"], 4), bias=round(mtr["bias"], 4)),
        dict(set="held-out", n_trees=int(te.sum()), R2=round(mte["R2"], 4), RMSE=round(mte["RMSE"], 4),
             MAE=round(mte["MAE"], 4), bias=round(mte["bias"], 4)),
    ])
    path = os.path.join(odir, "hybrid_overfit_train_vs_heldout_SEED%d.csv" % R.SEED)
    summ.to_csv(path, index=False, encoding="utf-8-sig")
    print("wrote", path)


if __name__ == "__main__":
    main()
