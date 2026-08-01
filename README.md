# CBH-CR hybrid model (Korea NFI)

Predicts **crown base height (CBH)** and **crown ratio (CR)** for tree species in the Republic of Korea
from National Forest Inventory (NFI) data, and produces nationwide **5 m CBH/CR maps**. The model is a
hybrid of an allometric baseline (Hasenauer–Monserud logistic CR) and an XGBoost residual/stacking layer.

> Status: revision for *Ecological Informatics* (ECOINF-D-25-04526). All performance is reported on
> **plot-disjoint** cross-validation (see below).

## Repository layout
- `src/` — analysis notebooks (`NFI-Preprocessing` → `EDA` → `Baseline` → `CRModel*` hybrid →
  `Evaluation` → `CRModel3-Mapping`) and scripts. Key reproducible scripts:
  - `src/wp0_plot_disjoint_retrain.py` — baseline + XGB hybrid with tree-vs-plot-disjoint splits.
  - `src/wp0_build_table2.py` — assembles the corrected Table 2.
  - `src/SamplingImsang.py` — class-conditional raster sampling for mapping.
- `result/` — metric tables and aggregate outputs (no per-tree NFI data; large rasters excluded).
- `paper/` — manuscript-facing text (Methods, reviewer responses).
- `test/` — small reproducibility slice (aggregate result tables; **input data not included**, see below).

## Environment & setup
Python 3.9 + a geospatial/ML stack. Two pinned conda envs are provided:
```bash
conda env create -f geo-ml3.yml   # use this for CLI model fitting
conda env create -f geo-ml4.yml   # geospatial/mapping
```
> **Reproducibility note:** run the CLI fitting scripts in **`geo-ml3`**. In `geo-ml4` (numpy 2.0 build),
> `scipy.optimize.curve_fit` can hard-crash from the command line; if you must use it, set
> `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`. `requirements.txt` is a separate TensorFlow stack,
> not the geospatial env.

## Reproducing the plot-disjoint results
```bash
conda activate geo-ml3
python src/wp0_plot_disjoint_retrain.py --model baseline --mode tree   # original (leaky) baseline
python src/wp0_plot_disjoint_retrain.py --model baseline --mode plot   # plot-disjoint baseline
python src/wp0_plot_disjoint_retrain.py --model hybrid   --mode tree   # original (leaky) hybrid
python src/wp0_plot_disjoint_retrain.py --model hybrid   --mode plot   # plot-disjoint hybrid
python src/wp0_build_table2.py                                          # → Table2_model_performance.csv
```
Determinism: global `SEED = 100`; `curve_fit` is seeded with `p0 = [3, -1, 0.1, -1]` (the published
optimum's basin — the default `[1,1,1,1]` diverges for large species in this scipy build).

## Data partitioning and cross-validation
NFI trees are nested in plots (`SampleID`; 187,933 trees across 13,859 plots, ~13.6 trees/plot), so they
are not independent. All splits and folds are **plot-disjoint**: a 20% plot hold-out via
`GroupShuffleSplit` and 5-fold `StratifiedGroupKFold` (group = `SampleID`, stratify = NFI cycle), with an
explicit train ∩ test = ∅ assertion. Tree-level partitioning would leak plot-mates and inflate scores.

## Results & honest limitations
- Under plot-disjoint evaluation the hybrid's **median per-species R² is 0.11** (pooled 0.19); the earlier
  tree-level split reported 0.18 (pooled 0.25) — an inflation of ~40% from within-plot leakage.
- **Many species have low or non-positive R²** (6 of 36 are R² ≤ 0 under plot-disjoint CV). We report
  R², RMSE, MAE, and bias per species; results are **not** uniformly strong and should not be over-read.
- CBH is an **observed** NFI field (지하고); CR = (H − CBH)/H is the modeling target.

## Data availability
This code is openly available on GitHub (https://github.com/mongbleu/cbh-cr-hybrid-korea) and archived on
Zenodo — **DOI: [10.5281/zenodo.21730483](https://doi.org/10.5281/zenodo.21730483)** (concept DOI, always
resolves to the latest version).

**The NFI and NIFS Imsang data are NOT redistributed in this repository** (licensing/ownership), and the
multi-GB terrain rasters are archived separately. This repository contains **code, aggregate result
tables, and small project-defined reference tables only**.

To reproduce from raw data, obtain the inputs from their sources and place them per `src/NFI-Preprocessing`:
- **NFI (6th/7th)** tree records — Korea Forest Service / NIFS National Forest Inventory (request access).
- **NIFS Imsang** forest-type geodatabase (`imsang_nifs.gdb`) — National Institute of Forest Science.
- **Terrain** (DEM/Slope/Aspect, EPSG:5179) — national DEM source; reproject to EPSG:5179.

Cite the original data providers; do not redistribute the raw or cleaned NFI/Imsang data.

## Citation & license
See `CITATION.cff`. Code is released under the **MIT License** (`LICENSE`); the license covers code only,
not the data. Archived on Zenodo — concept DOI **10.5281/zenodo.21730483**
(this release, version 1: 10.5281/zenodo.21730484).
