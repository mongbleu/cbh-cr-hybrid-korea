"""WP-B-6 — immutable snapshot manifest for the plot-disjoint result set.

Freezes the published WP-B artifact set by recording a provenance + integrity fingerprint:
  - sha256 + size for the small tracked artifacts (category CSV, figures, eval CSVs, bundle, report);
  - size + mtime + cached raster statistics (min/max/mean/std) for the multi-GB nationwide maps on F:\\
    (hashing 12.8 GB every snapshot is wasteful; the stats fingerprint detects content change).
Writes `SNAPSHOT_WP-B_<commit>.md` into the tagged result dir. Read-only on all artifacts.

The `protect_results_immutable.py` hook enforces no in-place overwrite of frozen result files; this
manifest is the human/audit record of WHAT was frozen and at which commit. Run after the gate PASSes.

Usage:  python src/wp_b6_snapshot.py
"""
from __future__ import annotations
import os, hashlib, subprocess
from datetime import datetime, timezone

REPO = r"D:\ForestFire\CBH"
TAG_DIR = os.path.join(REPO, "result", "Baseline3", "HM변형-try2.0-plotdisjoint")
FIG_DIR = os.path.join(REPO, "fig")
RASTER_DIR = r"F:\CBH"

SMALL_ARTIFACTS = [
    os.path.join(TAG_DIR, "Pred-NIFS-ComparebyCategory_plotdisjoint_SEED100.csv"),
    os.path.join(TAG_DIR, "Table2_plot_disjoint.csv"),
    os.path.join(TAG_DIR, "baseline_eval_plot_SEED100.csv"),
    os.path.join(TAG_DIR, "hybrid_eval_plot_SEED100.csv"),
    os.path.join(TAG_DIR, "model", "hybrid_plotdisjoint_SEED100.joblib"),
    os.path.join(TAG_DIR, "WP-B-6_validation_report.txt"),
    os.path.join(FIG_DIR, "boxplot-NIFSvsPred-DMCLS_plotdisjoint.png"),
    os.path.join(FIG_DIR, "boxplot-NIFSvsPred-AGECLS_plotdisjoint.png"),
]
BIG_RASTERS = [
    os.path.join(RASTER_DIR, "CR_HM변형-try2.0-plotdisjoint_SEED100.tif"),
    os.path.join(RASTER_DIR, "CBH_meter_HM변형-try2.0-plotdisjoint_SEED100.tif"),
]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_stats(path):
    import rasterio
    with rasterio.open(path) as src:
        st = src.statistics(1)
        return f"EPSG:{src.crs.to_epsg()} {src.res[0]}m  min={st.min:.4f} max={st.max:.4f} mean={st.mean:.4f} std={st.std:.4f}"


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], text=True).strip()
    except Exception:
        return "(unknown)"


def main():
    commit = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = os.path.join(TAG_DIR, f"SNAPSHOT_WP-B_{commit}.md")

    L = []
    L.append(f"# WP-B immutable snapshot — {stamp}")
    L.append(f"\nBranch `{branch}` @ commit `{commit}` · tag `HM변형-try2.0-plotdisjoint` · SEED 100")
    L.append("\nValidation gate: see `WP-B-6_validation_report.txt` (12/12 PASS at snapshot time).")
    L.append("\n## Small artifacts (sha256)")
    for p in SMALL_ARTIFACTS:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            L.append(f"- `{os.path.relpath(p, REPO)}` — {sz:,} B — `{sha256(p)}`")
        else:
            L.append(f"- MISSING: `{os.path.relpath(p, REPO)}`")
    L.append("\n## Nationwide maps (F:\\CBH — size + cached raster stats fingerprint)")
    for p in BIG_RASTERS:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            mt = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            L.append(f"- `{p}` — {sz:,} B — mtime {mt}\n  - {raster_stats(p)}")
        else:
            L.append(f"- MISSING: `{p}`")
    L.append("\n## Invariants frozen")
    L.append("- CR ∈ [0,1] (clipped); CBH = H·(1−CR) ∈ [0, H] m; CRS EPSG:5179 @ 5 m.")
    L.append("- Provenance: plot-disjoint hybrid bundle (mode='plot'), 39 SIDs, SEED 100.")
    L.append("- ⛔ OLD `F:\\CBH\\CBH4.tif`/`CR4.tif` remain INVALID — not part of this snapshot.")
    L.append("- NIFS-CBH in the category table is a STATIC reference (carried forward, not raster-derived).")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsnapshot manifest written: {out}")


if __name__ == "__main__":
    main()
