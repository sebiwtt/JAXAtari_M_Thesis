# =============================================================================
# Plot the CRL learning/forgetting curve (crl_curve_* arrays in matrix.npz)
# =============================================================================
# One continuous base-env return curve across the whole task sequence - the
# Dohare-style "loss of plasticity" figure. Two modes, chosen automatically
# from the cl_method recorded in each run's matrix.json:
#
#   single method   runs are seed replicates; ONE curve, colored per task
#                   segment, mean +- band when several seeds are given.
#   method compare  runs span several CL methods (e.g. ft/ewc/agem/packnet on
#                   the same sequence); one curve per method in a fixed method
#                   color, task segments marked by boundary lines + top labels.
#
#   python tools/plot_crl_curve.py runs/pong_ft_dyn4_oc_{0,1,2}/matrix.npz
#   python tools/plot_crl_curve.py runs/pong_{ft,ewc,agem,packnet}_dyn4_oc_*/matrix.npz \
#          --smooth 5 --out crl_methods.pdf
#
# Data contract (written by ppo_crl_continual when CRL_CURVE is on):
#   crl_curve_wandb_step  shared x-grid across tasks (PPO iterations, cumulative)
#   crl_curve_env_step    wandb_step * BATCH_SIZE - the axis plotted here
#   crl_curve_task_idx    0 = base, 1..N = mod tasks (segment indicator)
#   crl_curve_task_label  e.g. "base", "lazy_enemy"
#   crl_curve_source      "train" (base task's training return) | "eval"
#   crl_curve_base_return raw per-episode return on the base env
#   crl_curve_completed_frac  < 1 marks eval points with truncated episodes
# =============================================================================

import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# Categorical palette (validated: CVD-safe adjacent order on white; sub-3:1
# slots are relieved by the legend / direct labels).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Fixed slot per method so colors stay stable across figures ("color follows
# the entity"); methods not listed take the next free slots in file order.
METHOD_SLOTS = {"ft": 0, "ewc": 1, "agem": 2, "packnet": 3}


def load_curve(path: str) -> dict:
    """Read the crl_curve_* arrays from one matrix.npz (or a run dir containing it)."""
    if os.path.isdir(path):
        path = os.path.join(path, "matrix.npz")
    d = np.load(path)
    if "crl_curve_wandb_step" not in d.files:
        raise SystemExit(f"{path}: no crl_curve_* arrays - was this run made with CRL_CURVE=True?")
    curve = {k.removeprefix("crl_curve_"): d[k] for k in d.files if k.startswith("crl_curve_")}
    curve["method"] = _run_method(path, d)
    return curve


def _run_method(npz_path: str, d) -> str:
    json_path = os.path.join(os.path.dirname(npz_path), "matrix.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            method = json.load(f).get("cl_method")
        if method:
            return str(method)
    return str(d["exp_name"]) if "exp_name" in d.files else os.path.basename(npz_path)


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


def aggregate(group: list[dict], smooth_window: int, band: str):
    """Align one group of seed replicates and reduce to (x, meta, mean, spread)."""
    grid, meta, returns = align_on_grid(group)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices = warm-up gaps
        mean = smooth(np.nanmean(returns, axis=0), smooth_window)
        spread = np.nanstd(returns, axis=0)
    if band == "sem":
        n_seeds = (~np.isnan(returns)).sum(axis=0)
        spread = spread / np.sqrt(np.maximum(n_seeds, 1))
    return meta["env_step"] / 1e6, meta, mean, spread


def draw_line(ax, x, mean, spread, seg, color, label, show_band, meta):
    """One curve piece + band + eval markers/truncation rings, house style."""
    if show_band:
        ax.fill_between(x[seg], (mean - spread)[seg], (mean + spread)[seg], color=color, alpha=0.15, linewidth=0)
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


def draw_task_boundaries(ax, x, task_idx, task_label, annotate: bool):
    """Hairline verticals where task_idx increments; optional top segment labels."""
    for b in np.where(np.diff(task_idx) > 0)[0]:
        ax.axvline(x[b], color=GRID, linewidth=0.8, zorder=0)
    if annotate:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for t in np.unique(task_idx):
            seg = np.where(task_idx == t)[0]
            ax.text(x[seg].mean(), 1.01, str(task_label[seg[0]]), transform=trans,
                    ha="center", va="bottom", color=MUTED, fontsize=8)


def style_axes(ax, title: str):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=26)
    ax.set_xlabel("Environment steps (millions)", color=INK_2, fontsize=9)
    ax.set_ylabel("Return per episode (base env, raw)", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best", title=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="matrix.npz files (or run dirs); mixed CL methods switch to comparison mode")
    ap.add_argument("--out", default="crl_curve.png", help="output image (.png/.pdf/.svg)")
    ap.add_argument("--smooth", type=int, default=1, help="rolling-mean window in curve points (default off)")
    ap.add_argument("--band", choices=["std", "sem", "none"], default="std", help="spread band across seeds")
    ap.add_argument("--title", default=None, help="figure title (default depends on mode)")
    args = ap.parse_args()

    curves = [load_curve(p) for p in args.runs]
    methods = list(dict.fromkeys(c["method"] for c in curves))  # unique, file order

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    truncated_total = 0

    if len(methods) == 1:
        # --- seed-replicate mode: one curve, task segments carry the color ------
        x, meta, mean, spread = aggregate(curves, args.smooth, args.band)
        task_idx = meta["task_idx"].astype(int)
        for t in np.unique(task_idx):
            seg = np.where(task_idx == t)[0]
            label = str(meta["task_label"][seg[0]])
            if seg[0] > 0:  # bridge to the previous segment so the curve reads as one line
                seg = np.concatenate([[seg[0] - 1], seg])
            draw_line(ax, x, mean, spread, seg, SERIES[t % len(SERIES)], label,
                      show_band=len(curves) > 1 and args.band != "none", meta=meta)
        draw_task_boundaries(ax, x, task_idx, meta["task_label"], annotate=False)
        truncated_total = int(np.nansum(meta["completed_frac"] < 1.0))
        note_runs = f"{methods[0]}, {len(curves)} seed(s)"
        title = args.title or "Base-env return over the continual task sequence"
    else:
        # --- method-comparison mode: one curve per method, fixed method colors --
        free_slots = iter(i for i in range(len(SERIES)) if i not in METHOD_SLOTS.values())
        ref_labels = None
        for method in methods:
            group = [c for c in curves if c["method"] == method]
            x, meta, mean, spread = aggregate(group, args.smooth, args.band)
            slot = METHOD_SLOTS.get(method)
            color = SERIES[(slot if slot is not None else next(free_slots)) % len(SERIES)]
            draw_line(ax, x, mean, spread, np.arange(x.size), color, method,
                      show_band=len(group) > 1 and args.band != "none", meta=meta)
            truncated_total += int(np.nansum(meta["completed_frac"] < 1.0))
            labels_seq = [str(meta["task_label"][i]) for i in np.flatnonzero(np.r_[1, np.diff(meta["task_idx"].astype(int))])]
            if ref_labels is None:
                ref_labels, ref = labels_seq, (x, meta)
            elif labels_seq != ref_labels:
                print(f"WARNING: {method} runs have task sequence {labels_seq}, others {ref_labels} - comparison may be invalid.")
        x, meta = ref
        draw_task_boundaries(ax, x, meta["task_idx"].astype(int), meta["task_label"], annotate=True)
        counts = {m: sum(c["method"] == m for c in curves) for m in methods}
        note_runs = ", ".join(f"{m} n={n}" for m, n in counts.items())
        title = args.title or "Base-env return: CL methods compared"

    style_axes(ax, title)
    note = note_runs + ("" if args.band == "none" or len(curves) == 1 else f" (band = {args.band})")
    ax.text(1.0, 1.07, note, transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}  ({len(curves)} run(s): {note_runs})")
    if truncated_total:
        print(f"note: {truncated_total} eval point(s) had truncated episodes (completed_frac < 1) - drawn as hollow rings.")


if __name__ == "__main__":
    main()
