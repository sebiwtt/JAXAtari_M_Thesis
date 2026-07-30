# =============================================================================
# Plot the CRL learning/forgetting curve (crl_curve_* arrays in matrix.npz)
# =============================================================================
# One continuous base-env return curve across the whole task sequence, colored
# per task segment - the Dohare-style "loss of plasticity" figure. Pass several
# matrix.npz files (seed replicates of the same config) to get a mean line with
# a +- std band; a single file plots as-is.
#
#   python tools/plot_crl_curve.py runs/pong_ft_dyn4_oc_0/matrix.npz
#   python tools/plot_crl_curve.py runs/pong_ft_dyn4_oc_{0,1,2}/matrix.npz \
#          --smooth 5 --out crl_curve.pdf
#
# Data contract (written by ppo_crl_continual when CRL_CURVE is on):
#   crl_curve_wandb_step  shared x-grid across tasks (PPO iterations, cumulative)
#   crl_curve_env_step    wandb_step * BATCH_SIZE - the axis plotted here
#   crl_curve_task_idx    0 = base, 1..N = mod tasks (colors the segments)
#   crl_curve_task_label  e.g. "base", "lazy_enemy" (legend / direct labels)
#   crl_curve_source      "train" (base task's training return) | "eval"
#   crl_curve_base_return raw per-episode return on the base env
#   crl_curve_completed_frac  < 1 marks eval points with truncated episodes
# =============================================================================

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Categorical palette (validated: CVD-safe adjacent order on white; sub-3:1
# slots are relieved by the legend + direct labels). Task i -> SERIES[i % 8].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load_curve(path: str) -> dict:
    """Read the crl_curve_* arrays from one matrix.npz (or a run dir containing it)."""
    if os.path.isdir(path):
        path = os.path.join(path, "matrix.npz")
    d = np.load(path)
    if "crl_curve_wandb_step" not in d.files:
        raise SystemExit(f"{path}: no crl_curve_* arrays - was this run made with CRL_CURVE=True?")
    return {k.removeprefix("crl_curve_"): d[k] for k in d.files if k.startswith("crl_curve_")}


def align_on_grid(curves: list[dict]) -> tuple[np.ndarray, dict, np.ndarray]:
    """Union wandb_step grid across seeds; per-seed returns as rows (NaN off-grid).

    Seed replicates of one config share the grid exactly; the union only matters
    if a run was cut short. Segment metadata (env_step/task/source/completed) is
    taken from the first run that has each grid point.
    """
    grid = np.unique(np.concatenate([c["wandb_step"] for c in curves]))
    returns = np.full((len(curves), grid.size), np.nan)
    meta_src = np.full(grid.size, -1)
    meta = {k: None for k in ("env_step", "task_idx", "task_label", "source", "completed_frac")}
    for si, c in enumerate(curves):
        pos = np.searchsorted(grid, c["wandb_step"])
        returns[si, pos] = c["base_return"]
        new = meta_src[pos] == -1
        meta_src[pos[new]] = si
        for k in meta:
            if meta[k] is None:
                meta[k] = np.empty(grid.size, dtype=c[k].dtype)
            meta[k][pos[new]] = c[k][new]
    assert (meta_src >= 0).all()
    # Conservative across seeds: a point is flagged if ANY seed had truncated episodes.
    # (All-NaN slices - train points, warm-up gaps - are legitimate; keep them NaN quietly.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        meta["completed_frac"] = np.nanmin(np.stack([_on_grid(c, grid, "completed_frac") for c in curves]), axis=0)
    return grid, meta, returns


def _on_grid(c: dict, grid: np.ndarray, key: str) -> np.ndarray:
    out = np.full(grid.size, np.nan)
    out[np.searchsorted(grid, c["wandb_step"])] = c[key]
    return out


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered rolling mean (window=1 -> unchanged)."""
    if window <= 1:
        return y
    kernel = np.ones(window)
    mask = ~np.isnan(y)
    num = np.convolve(np.where(mask, y, 0.0), kernel, mode="same")
    den = np.convolve(mask.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore"):
        out = num / den
    out[~mask] = np.nan  # keep gaps (e.g. NAN_UNTIL_FIRST_EPISODE warm-up) as gaps
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="matrix.npz files (or run dirs) - several = seed replicates")
    ap.add_argument("--out", default="crl_curve.png", help="output image (.png/.pdf/.svg)")
    ap.add_argument("--smooth", type=int, default=1, help="rolling-mean window in curve points (default off)")
    ap.add_argument("--band", choices=["std", "sem", "none"], default="std", help="spread band across seeds")
    ap.add_argument("--title", default="Base-env return over the continual task sequence")
    args = ap.parse_args()

    curves = [load_curve(p) for p in args.runs]
    grid, meta, returns = align_on_grid(curves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices = warm-up gaps
        mean = smooth(np.nanmean(returns, axis=0), args.smooth)
        n_seeds = (~np.isnan(returns)).sum(axis=0)
        spread = np.nanstd(returns, axis=0)
    if args.band == "sem":
        spread = spread / np.sqrt(np.maximum(n_seeds, 1))

    x = meta["env_step"] / 1e6
    task_idx = meta["task_idx"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    seen_labels = []
    for t in np.unique(task_idx):
        seg = np.where(task_idx == t)[0]
        # Bridge to the previous segment's last point so the curve reads as one line.
        if seg[0] > 0:
            seg = np.concatenate([[seg[0] - 1], seg])
        color = SERIES[t % len(SERIES)]
        label = str(meta["task_label"][np.where(task_idx == t)[0][0]])
        seen_labels.append((label, color))
        if len(curves) > 1 and args.band != "none":
            ax.fill_between(x[seg], (mean - spread)[seg], (mean + spread)[seg], color=color, alpha=0.18, linewidth=0)
        ax.plot(x[seg], mean[seg], color=color, linewidth=1.8, solid_capstyle="round", label=label)
        # Point markers only where the eval grid is sparse enough to read as points;
        # dense segments stay a clean line. Hollow ring = truncated episodes
        # (completed_frac < 1 in any seed), whose returns are unreliable.
        is_eval = (meta["source"][seg] == "eval") if meta["source"].dtype.kind == "U" else np.zeros(len(seg), bool)
        if 0 < is_eval.sum() <= 25:
            ax.plot(x[seg][is_eval], mean[seg][is_eval], "o", color=color, markersize=3.5)
        truncated = is_eval & (meta["completed_frac"][seg] < 1.0)
        if truncated.any():
            ax.plot(x[seg][truncated], mean[seg][truncated], "o", markersize=6.5,
                    markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.0)

    # Task boundaries: hairline verticals where task_idx increments.
    for b in np.where(np.diff(task_idx) > 0)[0]:
        ax.axvline(x[b], color=GRID, linewidth=0.8, zorder=0)

    ax.set_title(args.title, color=INK, fontsize=11, loc="left", pad=12)
    ax.set_xlabel("Environment steps (millions)", color=INK_2, fontsize=9)
    ax.set_ylabel("Return per episode (base env, raw)", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best", title=None)

    if len(curves) > 1:
        note = f"mean over {len(curves)} seeds" + ("" if args.band == "none" else f" (band = {args.band})")
        ax.text(1.0, 1.02, note, transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}  ({grid.size} curve points, {len(curves)} run(s))")
    bad = int(np.nansum(meta["completed_frac"] < 1.0))
    if bad:
        print(f"note: {bad} eval point(s) had truncated episodes (completed_frac < 1) - drawn as hollow rings.")


if __name__ == "__main__":
    main()
