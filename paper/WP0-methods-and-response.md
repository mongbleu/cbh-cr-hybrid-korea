# WP-0 — Methods text & reviewer responses (plot-disjoint CV)

Ready-to-paste manuscript text and response-letter entries from the WP-0 re-train. All numbers come from
`result/Baseline3/HM변형-try2.0-plotdisjoint/` (script: `src/wp0_plot_disjoint_retrain.py`, SEED=100,
env `geo-ml3`). Evidence log: `memory/decisions/2026-06-21_wp0-cv-integrity-verdict.md`.

---

## Manuscript — new Methods subsection: "Data partitioning and cross-validation"

National Forest Inventory (NFI) trees are nested within sample plots (`SampleID`, the NFI 표본점 number):
our data comprise 187,933 trees across 13,859 plots (mean 13.6 trees/plot). Because trees in a plot share
stand and site conditions, they are not independent; partitioning at the tree level would place plot-mates
in both training and test sets and inflate apparent skill. We therefore partition **at the plot level**.
A held-out test set was formed with `GroupShuffleSplit` (20% of plots), and model selection used
`StratifiedGroupKFold` (5 folds; grouping = `SampleID`, stratification = NFI cycle), so that all trees of a
plot fall entirely within a single fold and the training and test plot sets are disjoint
(verified: train ∩ test = ∅). The allometric baseline (Hasenauer–Monserud logistic crown-ratio model) and
the XGBoost residual/stacking layer were both fit on training plots only and evaluated on held-out plots.
Crown base height (CBH) is an observed NFI field (지하고); crown ratio CR = (H − CBH)/H is the modeling target.
Performance is reported per species and pooled as R², RMSE, MAE, and bias, with cross-validation variance.

## Effect of plot-disjoint evaluation (replaces inflated tree-level numbers)

| Model | metric | tree-level (original) | plot-disjoint (corrected) |
|---|---|---|---|
| Hybrid (XGBoost) | median per-species R² | 0.182 | **0.107** (−41%) |
| Hybrid (XGBoost) | pooled R² | 0.250 | **0.192** |
| Hybrid (XGBoost) | species with R² ≤ 0 (of 36) | 1 | **6** |
| Hybrid (XGBoost) | overfitting gap (test−train RMSE) | 0.005 | **0.012** |
| Allometric baseline | median per-species R² | 0.074 | 0.043 |

The inflation is largest for the hybrid because its spatial covariates (latitude, longitude, elevation,
slope, aspect) are near-constant within a plot; tree-level splitting leaks them. Inflation is most severe
for species sampled in few plots and modest for well-sampled species.

**Species counts (36 vs 39).** The plot-disjoint *evaluation* covers the **36 species** that retained
enough held-out plots to yield stable per-species metrics. The nationwide CBH/CR *maps* (WP-B) cover
**39 species**: the additional low-sample species are retained by borrowing a substitute species'
allometric parameters (`Alternative_Species_Model.csv`) rather than being dropped. The two numbers refer
to different stages (evaluation vs. mapping) and are not in conflict.

**Nationwide map bias (honest reporting).** Validated against the NIFS reference CBH by DBH and age class,
the mapped predictions read **+0.61 m high on average** (pooled bias +0.61 m, MAE 1.34 m, RMSE 1.72 m over
114 categories). The positive bias is largest in the smallest-DBH class (+1.32 m) and the oldest/youngest
age classes (age class 7 +1.58 m, age class 2 +1.21 m). We report this bias explicitly and do not
characterise the maps as accurate where they are not.

---

## Response letter

**R2-3 (independence / data leakage).** We agree. The original train/test partition and cross-validation
operated at the tree level, so trees from the same plot could appear in both training and test sets. We
re-implemented the full pipeline with **plot-disjoint** partitioning (group = NFI plot `SampleID`):
`GroupShuffleSplit` for the hold-out and `StratifiedGroupKFold` for model selection, with an explicit
assertion that training and test plot sets are disjoint. Under this corrected protocol the hybrid model's
median per-species R² is 0.11 (vs. 0.18 under the leaky split) and pooled R² is 0.19 (vs. 0.25); six of
36 species are non-predictive (R² ≤ 0). All reported metrics now derive from plot-disjoint data. *(Methods
§ "Data partitioning and cross-validation"; Table 2 regenerated.)*

**R2-2 / R2-8 (split description).** A dedicated Methods subsection now defines the partition unit (plot),
the hold-out scheme (20% of plots via `GroupShuffleSplit`), the fold count and stratification
(5-fold `StratifiedGroupKFold`, grouped by plot, stratified by NFI cycle), and the disjointness guarantee.

**R1-2 (overfitting).** We captured XGBoost `evals_result_` (train vs. validation RMSE). Under the
corrected plot-disjoint evaluation the train–test RMSE gap is ~0.012 (vs. ~0.005 under the leaky split):
honest partitioning reveals more of the optimism the original protocol masked. We report this curve and the
fold variance in the supplement.

**R2-9 (zero R² values).** The previously tabulated R² = 0 entries were **negative R² clipped to zero**
(`clip(lower=0)` in the table-build step). We now report the unclipped values; several low-n species have
genuinely negative R², which we state plainly rather than masking.

**R2-10 (Table 2 n = 0.594).** This was a column/merge misalignment in the table build. The regenerated
Table 2 derives the per-species sample size directly from the cleaned data, so every `n` is an integer
(`Table2_plot_disjoint.csv`, `n_total` column).

**Honest performance language / map validation.** We have removed generic "satisfactory" claims and now
report bias alongside R²/RMSE/MAE throughout. The nationwide maps, regenerated from the plot-disjoint
model, were validated against the NIFS reference CBH by DBH and age class: they read **+0.61 m high on
average** (MAE 1.34 m, RMSE 1.72 m), with the largest positive bias in the smallest-DBH and oldest-age
classes. We state this bias in the text rather than presenting the maps as unbiased. (The 36 species used
for per-species evaluation vs. the 39 species carried into the maps are distinguished in Methods; the extra
species use substitute-species allometry rather than being dropped.)

---

## Downstream (flag for W2/W3)
- WP-B: maps and box-plots must be regenerated from the plot-disjoint model.
- WP-D: ship `src/wp0_plot_disjoint_retrain.py` + the reproducibility notes (seeded `curve_fit` `p0`,
  `geo-ml3` for CLI scipy) and `Table2_plot_disjoint.csv` in the public repo.
