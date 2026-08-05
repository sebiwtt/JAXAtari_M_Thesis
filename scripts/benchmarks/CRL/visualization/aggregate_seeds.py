# =============================================================================
# Aggregate seed replicates of a CRL run into a single file
# =============================================================================
# Collapses runs/<group>_1, <group>_2, <group>_3 (the seed replicates written by
# ppo_crl_continual.py) into runs/aggregated/<group>/aggregate.{json,npz}.
#
# What is aggregated, and how:
#
#   Every quantity is reduced ACROSS SEEDS per element (per matrix cell, per task,
#   per curve point). For each one we keep mean / std / sem / ci95 / median / min /
#   max / n *and the raw per-seed values*, so nothing downstream is locked into one
#   summary statistic. Bookkeeping:
#
#     std   sample std, ddof=1 (numpy's default ddof=0 would understate spread and
#           is simply the wrong estimator for 3 draws from a seed distribution)
#     sem   std / sqrt(n) - uncertainty OF THE MEAN; use this for error bars when
#           the claim is "method A beats method B", not "seeds vary a lot"
#     ci95  half-width of the Student-t 95% interval, t(0.975, n-1) * sem. At n=3
#           t = 4.303, so the interval is ~2.2x the sem - wide, but honest.
#     n     per-element count of non-NaN seeds. Not always 3: a seed whose base
#           task collapsed leaves Retention[:, 0] undefined (R[j,j] <= R_rand[j]),
#           so that column aggregates over fewer seeds. Plots should surface n < 3.
#
#   Derived metrics (Retention, Drop, Forgetting, mean_forgetting, ...) are
#   aggregated PER SEED and then averaged - they are NOT recomputed from the mean R.
#   Retention is a clipped ratio, so mean-of-ratios != ratio-of-means, and the seed
#   is the sampling unit whose variability the error bars are meant to describe.
#   R and R_rand are averaged too, so raw-return figures stay available, but the
#   Retention block is the authoritative one.
#
# Usage:
#   python visualization/aggregate_seeds.py                       # every group under runs/
#   python visualization/aggregate_seeds.py runs/pong_ft_dyn4_oc_*
#   python visualization/aggregate_seeds.py --out-root /tmp/agg --min-seeds 2
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = SCRIPT_DIR.parent / "runs"
DEFAULT_OUT_DIRNAME = "aggregated"

# runs/<group>_<seed>; seed is the trailing integer of the dir name.
SEED_RE = re.compile(r"^(?P<group>.+)_(?P<seed>\d+)$")

# Two-sided 95% Student-t quantiles, t(0.975, df). Inlined so this stays a
# numpy-only script; df > 30 falls back to the normal quantile.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

# Matrix/vector quantities carried straight over from each run's matrix.json.
MATRIX_KEYS = ("R", "R_rand", "Retention", "Drop", "Forgetting")
# Scalars written by the harness, plus the derived ones computed here.
SCALAR_KEYS = ("mean_forgetting", "mean_retention")
CURVE_META = ("wandb_step", "env_step", "task_idx", "task_label", "source")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_run(run_dir: str | Path) -> dict:
    """Read one run's matrix.json (npz fallback) into plain numpy/py types."""
    run_dir = Path(run_dir)
    json_path, npz_path = run_dir / "matrix.json", run_dir / "matrix.npz"

    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)
        out = {k: np.asarray(d[k], dtype=float) for k in MATRIX_KEYS}
        out.update({k: float(d[k]) for k in SCALAR_KEYS})
        out["labels"] = list(d["labels"])
        out["task_mods"] = [list(m) for m in d["task_mods"]]
        out["task_timesteps"] = [int(t) for t in d["task_timesteps"]]
        out["env_id"] = d.get("env_id", "unknown")
        out["exp_name"] = d.get("exp_name", "unknown")
        out["cl_method"] = d.get("cl_method", "unknown")
        out["compute_time_sec"] = d.get("compute_time_sec", {})
        curve = d.get("crl_curve")
        out["crl_curve"] = (
            {k: np.asarray(v) for k, v in curve.items()} if curve else None
        )
    elif npz_path.exists():
        # Older/curve-only runs: same fields, minus what the npz never stored.
        z = np.load(npz_path, allow_pickle=True)
        out = {k: z[k].astype(float) for k in MATRIX_KEYS}
        out.update({k: float(z[k]) for k in SCALAR_KEYS})
        out["labels"] = [str(l) for l in z["labels"]]
        out["task_mods"] = [json.loads(str(m)) for m in z["task_mods"]]
        out["task_timesteps"] = [int(t) for t in z["task_timesteps"]]
        out["env_id"] = str(z["env_id"]) if "env_id" in z.files else "unknown"
        out["exp_name"] = str(z["exp_name"]) if "exp_name" in z.files else "unknown"
        out["cl_method"] = "unknown"
        out["compute_time_sec"] = {"total": float(z["total_compute_time"])} if "total_compute_time" in z.files else {}
        curve_keys = [k for k in z.files if k.startswith("crl_curve_")]
        out["crl_curve"] = (
            {k.removeprefix("crl_curve_"): z[k] for k in curve_keys} if curve_keys else None
        )
    else:
        raise FileNotFoundError(f"no matrix.json or matrix.npz in {run_dir}")

    out["run_dir"] = str(run_dir)
    return out


def discover_groups(runs_root: Path, out_root: Path) -> dict[str, list[Path]]:
    """Map <group> -> [run dirs], from every runs/<group>_<seed> holding a matrix."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir() or entry.resolve() == out_root.resolve():
            continue
        if not ((entry / "matrix.json").exists() or (entry / "matrix.npz").exists()):
            continue
        m = SEED_RE.match(entry.name)
        if m is None:
            print(f"[agg] skipping {entry.name}: no trailing _<seed> in the name")
            continue
        groups[m.group("group")].append(entry)
    return dict(groups)


def group_of(run_dir: Path) -> tuple[str, int]:
    m = SEED_RE.match(run_dir.name)
    if m is None:
        raise SystemExit(f"{run_dir}: expected a run dir named <group>_<seed>")
    return m.group("group"), int(m.group("seed"))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def summarize(stack: np.ndarray) -> dict:
    """Reduce a (n_seeds, ...) stack across axis 0 into the stat bundle.

    NaN-aware throughout: a cell undefined in one seed (e.g. Retention where that
    seed's R[j,j] <= R_rand[j]) is dropped from that cell only, and `n` records how
    many seeds actually backed each element. std/sem/ci95 need n >= 2 and are NaN
    below that - which is the truthful answer, not a bug to paper over.
    """
    stack = np.asarray(stack, dtype=float)
    n = np.sum(~np.isnan(stack), axis=0).astype(int)
    with warnings.catch_warnings():
        # All-NaN slices (undefined cells) and ddof >= count are both expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=1)
        median = np.nanmedian(stack, axis=0)
        lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
    std = np.where(n >= 2, std, np.nan)
    sem = std / np.sqrt(np.maximum(n, 1))
    tcrit = np.array([T95.get(int(k) - 1, 1.960) for k in np.atleast_1d(n).ravel()]).reshape(np.shape(n))
    ci95 = np.where(n >= 2, tcrit, np.nan) * sem
    return {
        "mean": mean, "std": std, "sem": sem, "ci95": ci95,
        "median": median, "min": lo, "max": hi, "n": n, "seeds": stack,
    }


def derived_metrics(run: dict) -> dict[str, float]:
    """Per-seed CL summary scalars (mirrors tools/visualize_matrix.compute_metrics).

    Computed per seed so they can be aggregated like any other measurement, rather
    than being read off the averaged matrices.
    """
    R, Retention = run["R"], run["Retention"]
    n = R.shape[0]
    diag = np.diag(R)
    lower = [Retention[i, j] for i in range(n) for j in range(i) if np.isfinite(Retention[i, j])]
    # Backward transfer: final performance on an earlier task vs. right after learning it.
    bwt = [R[n - 1, j] - diag[j] for j in range(n - 1)
           if np.isfinite(R[n - 1, j]) and np.isfinite(diag[j])]
    # Unclipped twin of final_avg_retention (so >1, i.e. positive backward transfer,
    # stays visible). Same guard as the harness: a task whose own post-training score
    # never beat the random floor has no meaningful scale, so it drops out rather than
    # dividing by a negative denominator.
    denom = diag - run["R_rand"]
    norm = np.where(denom > 0, (R[n - 1, :] - run["R_rand"]) / np.where(denom > 0, denom, 1.0), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "final_avg_return": float(np.nanmean(R[n - 1, :])),
            "final_avg_retention": float(np.nanmean(Retention[n - 1, :])),
            "avg_retention_lower": float(np.mean(lower)) if lower else float("nan"),
            "backward_transfer": float(np.mean(bwt)) if bwt else float("nan"),
            "avg_final_return_norm": float(np.nanmean(norm)),
            "total_compute_time_sec": float(run["compute_time_sec"].get("total", np.nan)),
        }


def align_curves(runs: list[dict]) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """Union wandb_step grid across seeds; per-seed base_return as rows (NaN off-grid).

    Seed replicates of one config share the grid exactly (verified for the pong
    sweep); the union only matters if a run was cut short, in which case the
    missing points simply lower `n` there.
    """
    curves = [r["crl_curve"] for r in runs if r.get("crl_curve") is not None]
    if not curves:
        return None, None
    grid = np.unique(np.concatenate([c["wandb_step"].astype(np.int64) for c in curves]))
    returns = np.full((len(curves), grid.size), np.nan)
    completed = np.full((len(curves), grid.size), np.nan)
    meta: dict[str, np.ndarray] = {}
    filled = np.zeros(grid.size, dtype=bool)
    for si, c in enumerate(curves):
        pos = np.searchsorted(grid, c["wandb_step"].astype(np.int64))
        returns[si, pos] = np.asarray(c["base_return"], dtype=float)
        completed[si, pos] = np.asarray(c["completed_frac"], dtype=float)
        new = ~filled[pos]
        for k in CURVE_META:
            if k not in meta:
                meta[k] = np.empty(grid.size, dtype=np.asarray(c[k]).dtype)
            meta[k][pos[new]] = np.asarray(c[k])[new]
        filled[pos] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Conservative across seeds: flagged if ANY seed had truncated episodes.
        meta["completed_frac_min"] = np.nanmin(completed, axis=0)
    return meta, returns


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def aggregate_group(group: str, run_dirs: list[Path]) -> dict:
    runs = []
    for d in sorted(run_dirs, key=lambda p: group_of(p)[1]):
        r = load_run(d)
        r["seed"] = group_of(d)[1]
        runs.append(r)

    ref = runs[0]
    for r in runs[1:]:  # a silent mismatch here would average incomparable numbers
        for key in ("labels", "task_timesteps"):
            assert r[key] == ref[key], (
                f"{group}: {r['run_dir']} has {key}={r[key]}, {ref['run_dir']} has {ref[key]}"
            )
        if r["cl_method"] != ref["cl_method"]:
            print(f"[agg] WARNING: {group}: cl_method differs ({r['cl_method']} vs {ref['cl_method']})")

    stats = {k: summarize(np.stack([r[k] for r in runs])) for k in MATRIX_KEYS}
    stats.update({k: summarize(np.array([r[k] for r in runs])) for k in SCALAR_KEYS})
    per_seed_derived = [derived_metrics(r) for r in runs]
    stats.update({
        k: summarize(np.array([d[k] for d in per_seed_derived]))
        for k in per_seed_derived[0]
    })

    curve_meta, curve_returns = align_curves(runs)
    curve = None
    if curve_meta is not None:
        curve = {k: curve_meta[k] for k in CURVE_META}
        curve["completed_frac_min"] = curve_meta["completed_frac_min"]
        curve["base_return"] = summarize(curve_returns)

    return {
        "group": group,
        "env_id": ref["env_id"],
        "exp_name": ref["exp_name"],
        "cl_method": ref["cl_method"],
        "modality": "pixel" if group.endswith("_pixel") else "oc" if group.endswith("_oc") else "unknown",
        "labels": ref["labels"],
        "task_mods": ref["task_mods"],
        "task_timesteps": ref["task_timesteps"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "run_dirs": [r["run_dir"] for r in runs],
        "stats": stats,
        "crl_curve": curve,
    }


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def write_aggregate(agg: dict, out_dir: Path) -> None:
    """Write aggregate.json (canonical, matches matrix.json's NaN spelling) + .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(_jsonify(agg), f, indent=2)

    flat: dict[str, np.ndarray] = {}
    for name, bundle in agg["stats"].items():
        for stat, val in bundle.items():
            flat[f"{name}_{stat}"] = np.asarray(val)
    if agg["crl_curve"] is not None:
        for k in CURVE_META + ("completed_frac_min",):
            flat[f"crl_curve_{k}"] = np.asarray(agg["crl_curve"][k])
        for stat, val in agg["crl_curve"]["base_return"].items():
            flat[f"crl_curve_base_return_{stat}"] = np.asarray(val)
    np.savez(
        out_dir / "aggregate.npz",
        **flat,
        group=np.array(agg["group"]),
        env_id=np.array(agg["env_id"]),
        exp_name=np.array(agg["exp_name"]),
        cl_method=np.array(agg["cl_method"]),
        modality=np.array(agg["modality"]),
        labels=np.array(agg["labels"]),
        task_timesteps=np.array(agg["task_timesteps"]),
        seeds=np.array(agg["seeds"]),
        n_seeds=np.array(agg["n_seeds"]),
    )


def print_summary(agg: dict) -> None:
    s = agg["stats"]
    def fmt(key: str) -> str:
        b = s[key]
        return f"{float(b['mean']):.3f} +- {float(b['std']):.3f}"
    n_min = int(np.min(s["Retention"]["n"][np.tril_indices(len(agg["labels"]))]))
    warn = "" if n_min == agg["n_seeds"] else f"   [!] some Retention cells use only {n_min}/{agg['n_seeds']} seeds"
    print(
        f"  {agg['group']:<28s} n={agg['n_seeds']}  "
        f"forgetting {fmt('mean_forgetting')}   "
        f"final_return {fmt('final_avg_return')}{warn}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Average seed replicates of a CRL run into one file.")
    ap.add_argument("runs", nargs="*", type=Path,
                    help="run dirs (runs/<group>_<seed>); default: every group under --runs-root")
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                    help=f"directory holding the run dirs (default: {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--out-root", type=Path, default=None,
                    help=f"output root (default: <runs-root>/{DEFAULT_OUT_DIRNAME})")
    ap.add_argument("--min-seeds", type=int, default=2,
                    help="skip groups with fewer replicates than this (default: 2)")
    ap.add_argument("--quiet", action="store_true", help="only print the output paths")
    args = ap.parse_args()

    out_root = args.out_root or args.runs_root / DEFAULT_OUT_DIRNAME

    if args.runs:
        groups: dict[str, list[Path]] = defaultdict(list)
        for d in args.runs:
            groups[group_of(d)[0]].append(d)
        groups = dict(groups)
    else:
        if not args.runs_root.is_dir():
            raise SystemExit(f"--runs-root {args.runs_root} is not a directory")
        groups = discover_groups(args.runs_root, out_root)

    if not groups:
        raise SystemExit("no runs found")

    written = 0
    for group, dirs in sorted(groups.items()):
        if len(dirs) < args.min_seeds:
            print(f"[agg] skipping {group}: only {len(dirs)} seed(s) < --min-seeds {args.min_seeds}")
            continue
        agg = aggregate_group(group, dirs)
        write_aggregate(agg, out_root / group)
        written += 1
        if not args.quiet:
            print_summary(agg)

    print(f"\n[agg] wrote {written} aggregate(s) to {out_root}{os.sep}<group>/aggregate.{{json,npz}}")


if __name__ == "__main__":
    main()
