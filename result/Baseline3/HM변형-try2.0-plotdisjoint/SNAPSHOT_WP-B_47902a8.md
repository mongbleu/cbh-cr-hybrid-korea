# WP-B immutable snapshot — 2026-06-23T11:16:38Z

Branch `revision/wp0-cv-integrity` @ commit `47902a8` · tag `HM변형-try2.0-plotdisjoint` · SEED 100

Validation gate: see `WP-B-6_validation_report.txt` (12/12 PASS at snapshot time).

## Small artifacts (sha256)
- `result\Baseline3\HM변형-try2.0-plotdisjoint\Pred-NIFS-ComparebyCategory_plotdisjoint_SEED100.csv` — 13,306 B — `d9683d1bab0998fe0ce33d2a938033618bdc494814ae41b5851973c9b6b4ea9e`
- `result\Baseline3\HM변형-try2.0-plotdisjoint\Table2_plot_disjoint.csv` — 3,699 B — `42fb3b0881851833c783477c34cdf3c1c8e99f711fb7ad57da5a16af346683a7`
- `result\Baseline3\HM변형-try2.0-plotdisjoint\baseline_eval_plot_SEED100.csv` — 2,221 B — `2d52c96e30d77554ef9594fd2b13dd4e426721ca5706655b26d9ca381b7c1a19`
- `result\Baseline3\HM변형-try2.0-plotdisjoint\hybrid_eval_plot_SEED100.csv` — 1,887 B — `066b8670ef804e573c130341fe121f9ede690b2b120c0871314777c22da6df2e`
- `result\Baseline3\HM변형-try2.0-plotdisjoint\model\hybrid_plotdisjoint_SEED100.joblib` — 486,842 B — `53ed286bee4c849a0b31fc2146da9e660ee700dfbf45d677bc36e046ac236f0b`
- `result\Baseline3\HM변형-try2.0-plotdisjoint\WP-B-6_validation_report.txt` — 2,037 B — `8bab7efe4469ef62beb42716453f7c57168fd66880adb32cea5219f286cf6a40`
- `fig\boxplot-NIFSvsPred-DMCLS_plotdisjoint.png` — 84,304 B — `018884a17cc052394e14ed6d35c8e4038ccecbbc3b335eddd77624090c7ccff9`
- `fig\boxplot-NIFSvsPred-AGECLS_plotdisjoint.png` — 99,542 B — `108118879056ca4c08b3797a04f5e5d480a3549db4722a600be28504d35f5b25`

## Nationwide maps (F:\CBH — size + cached raster stats fingerprint)
- `F:\CBH\CR_HM변형-try2.0-plotdisjoint_SEED100.tif` — 6,821,098,785 B — mtime 2026-06-22T15:21:09Z
  - EPSG:5179 5.0m  min=0.0312 max=1.0000 mean=0.5180 std=0.1008
- `F:\CBH\CBH_meter_HM변형-try2.0-plotdisjoint_SEED100.tif` — 6,844,189,104 B — mtime 2026-06-22T15:21:09Z
  - EPSG:5179 5.0m  min=0.0000 max=27.4305 mean=6.9206 std=2.7841

## Invariants frozen
- CR ∈ [0,1] (clipped); CBH = H·(1−CR) ∈ [0, H] m; CRS EPSG:5179 @ 5 m.
- Provenance: plot-disjoint hybrid bundle (mode='plot'), 39 SIDs, SEED 100.
- ⛔ OLD `F:\CBH\CBH4.tif`/`CR4.tif` remain INVALID — not part of this snapshot.
- NIFS-CBH in the category table is a STATIC reference (carried forward, not raster-derived).
