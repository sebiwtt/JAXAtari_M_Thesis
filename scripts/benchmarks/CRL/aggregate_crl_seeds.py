# =============================================================================
# Aggregate ppo_crl_continual.py results (see run_all_crl_seeds.py) across seed
# replicates: loads each runs/{...}_{SEED}/matrix.json, checks they share the same
# task sequence, and reports elementwise mean +/- std for R, R_rand, and Retention.
# =============================================================================

import argparse
import glob
import json
import os
import re

import numpy as np

from crl_benchmark import _print_matrix, _print_vector


def _load_matrix_jsons(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        matrix_path = p if p.endswith(".json") else os.path.join(p, "matrix.json")
        with open(matrix_path) as f:
            results.append(json.load(f))
    return results


def aggregate(paths: list[str]) -> dict:
    runs = _load_matrix_jsons(paths)
    assert len(runs) >= 2, f"Need at least 2 seed runs to aggregate meaningfully, got {len(runs)}"

    labels = runs[0]["labels"]
    for i, run in enumerate(runs[1:], start=1):
        assert run["labels"] == labels, (
            f"{paths[i]} has labels {run['labels']} but {paths[0]} has {labels} - "
            "seed replicates must share the same TASK_MODS sequence to be comparable."
        )

    R_stack = np.array([run["R"] for run in runs])  # (num_seeds, T, T)
    R_rand_stack = np.array([run["R_rand"] for run in runs])  # (num_seeds, T)
    Retention_stack = np.array([run["Retention"] for run in runs])  # (num_seeds, T, T)

    # Flag cells that are NaN in only some seeds (e.g. EVAL_FULL_MATRIX differed between
    # replicates) instead of silently averaging over fewer than num_seeds seeds.
    for name, stack in [("R", R_stack), ("Retention", Retention_stack)]:
        valid_counts = np.sum(~np.isnan(stack), axis=0)
        inconsistent = (valid_counts > 0) & (valid_counts < len(runs))
        if inconsistent.any():
            cells = list(zip(*np.where(inconsistent)))
            preview = cells[:10]
            print(
                f"[CRL] WARNING: {len(cells)} {name} cell(s) have inconsistent NaN patterns across seeds "
                f"(e.g. EVAL_FULL_MATRIX differed between runs) - their mean/std are computed over fewer "
                f"than {len(runs)} seeds: {preview}{'...' if len(cells) > len(preview) else ''}"
            )

    with np.errstate(invalid="ignore"):
        R_mean, R_std = np.nanmean(R_stack, axis=0), np.nanstd(R_stack, axis=0)
        R_rand_mean, R_rand_std = np.nanmean(R_rand_stack, axis=0), np.nanstd(R_rand_stack, axis=0)
        Retention_mean, Retention_std = np.nanmean(Retention_stack, axis=0), np.nanstd(Retention_stack, axis=0)

    return dict(
        labels=labels,
        task_mods=runs[0]["task_mods"],
        num_seeds=len(runs),
        R_mean=R_mean, R_std=R_std,
        R_rand_mean=R_rand_mean, R_rand_std=R_rand_std,
        Retention_mean=Retention_mean, Retention_std=Retention_std,
    )


def main():
    parser = argparse.ArgumentParser(description="Aggregate ppo_crl_continual.py matrix.json results across seed replicates.")
    parser.add_argument("--glob", type=str, default=None, help="Glob matching seed run dirs, e.g. 'runs/pong_ppo_crl_continual_oc_*'.")
    parser.add_argument("--runs", type=str, nargs="+", default=None, help="Explicit run dirs or matrix.json paths; overrides --glob.")
    parser.add_argument("--out", type=str, default=None, help="Output dir for aggregate.json/.npz. Defaults to runs/<common-prefix>_aggregate/.")
    args = parser.parse_args()

    if args.runs:
        paths = args.runs
    elif args.glob:
        paths = sorted(glob.glob(args.glob))
    else:
        parser.error("Provide either --runs <dir1> <dir2> ... or --glob '<pattern>'")

    assert paths, f"No runs found (glob={args.glob!r}, runs={args.runs!r})"
    print(f"Aggregating {len(paths)} seed run(s):")
    for p in paths:
        print(f"  {p}")

    result = aggregate(paths)
    labels = result["labels"]
    n = result["num_seeds"]

    _print_matrix(f"R mean (n={n} seeds)", result["R_mean"], labels)
    _print_matrix(f"R std (n={n} seeds)", result["R_std"], labels)
    _print_vector(f"R_rand mean (n={n} seeds)", result["R_rand_mean"], labels)
    _print_vector(f"R_rand std (n={n} seeds)", result["R_rand_std"], labels)
    _print_matrix(f"Retention mean (n={n} seeds)", result["Retention_mean"], labels)
    _print_matrix(f"Retention std (n={n} seeds)", result["Retention_std"], labels)

    if args.out:
        out_dir = args.out
    else:
        base = os.path.basename(paths[0].rstrip("/")).removesuffix(".json").removesuffix("/matrix")
        group_name = re.sub(r"_\d+$", "", base)
        out_dir = f"runs/{group_name}_aggregate"
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        f"{out_dir}/aggregate.npz",
        labels=np.array(labels),
        R_mean=result["R_mean"], R_std=result["R_std"],
        R_rand_mean=result["R_rand_mean"], R_rand_std=result["R_rand_std"],
        Retention_mean=result["Retention_mean"], Retention_std=result["Retention_std"],
    )
    with open(f"{out_dir}/aggregate.json", "w") as f:
        json.dump(
            {
                "num_seeds": n,
                "source_runs": paths,
                "labels": labels,
                "task_mods": result["task_mods"],
                "R_mean": result["R_mean"].tolist(),
                "R_std": result["R_std"].tolist(),
                "R_rand_mean": result["R_rand_mean"].tolist(),
                "R_rand_std": result["R_rand_std"].tolist(),
                "Retention_mean": result["Retention_mean"].tolist(),
                "Retention_std": result["Retention_std"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"\n[CRL] aggregate saved to {out_dir}/aggregate.npz and {out_dir}/aggregate.json")


if __name__ == "__main__":
    main()
