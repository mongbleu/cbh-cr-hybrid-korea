"""Strip outputs + execution counts from notebooks before public release (WP-D).

Notebook output cells can embed non-redistributable NFI data previews (df.head(), value tables) and
bloat the repo. This clears every code cell's `outputs` and `execution_count` in place, leaving code and
markdown untouched. Pure stdlib (no nbformat) so it runs in any environment; preserves non-ASCII (Korean).

Originals remain recoverable from git history. Usage:
    python src/strip_notebook_outputs.py            # strip src/*.ipynb (skips checkpoints)
    python src/strip_notebook_outputs.py --dry-run  # report only, write nothing
"""
import json, glob, os, argparse


def strip(path, dry):                                          # strip one notebook; return (cells_cleared, bytes_before, bytes_after)
    before = os.path.getsize(path)                             # size on disk before
    with open(path, encoding="utf-8") as f:                    # notebooks are UTF-8 JSON
        nb = json.load(f)
    cleared = 0                                                # count of code cells touched
    for cell in nb.get("cells", []):                           # walk every cell
        if cell.get("cell_type") == "code":                    # only code cells carry outputs
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cleared += 1                                   # this cell had output/exec state
            cell["outputs"] = []                               # drop rendered outputs (data previews, images)
            cell["execution_count"] = None                     # reset the run counter
            cell.get("metadata", {}).pop("execution", None)    # drop per-cell execution timing if present
    nb.get("metadata", {}).pop("widgets", None)                # drop bulky widget state if present
    if dry:                                                    # report-only mode
        return cleared, before, before
    with open(path, "w", encoding="utf-8") as f:               # write back, preserving Korean + jupyter style
        json.dump(nb, f, ensure_ascii=False, indent=1)         # indent=1 matches nbformat's default
        f.write("\n")                                          # trailing newline (nbformat convention)
    return cleared, before, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")          # preview without writing
    ap.add_argument("--glob", default="src/*.ipynb")           # which notebooks to process
    args = ap.parse_args()
    paths = [p for p in glob.glob(args.glob) if ".ipynb_checkpoints" not in p]  # skip checkpoint copies
    tot_b = tot_a = tot_c = 0                                   # running totals
    for p in sorted(paths):                                     # deterministic order
        c, b, a = strip(p, args.dry_run)                       # strip (or preview) this notebook
        tot_c += c; tot_b += b; tot_a += a                     # accumulate
        print(f"  {'(dry) ' if args.dry_run else ''}{os.path.basename(p):42s} cells_cleared={c:3d}  "
              f"{b/1e6:6.2f}MB -> {a/1e6:5.2f}MB")
    print(f"TOTAL: {len(paths)} notebooks, {tot_c} cells cleared, {tot_b/1e6:.1f}MB -> {tot_a/1e6:.1f}MB")


if __name__ == "__main__":
    main()
