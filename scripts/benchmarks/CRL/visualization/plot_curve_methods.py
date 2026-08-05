# =============================================================================
# Base-env return over a task sequence: all CL methods on one axes
# =============================================================================
# The aggregate-fed successor to tools/plot_crl_curve.py's comparison mode. It
# reads the seed-averaged aggregate.json files (aggregate_seeds.py) instead of
# re-reducing per-seed matrix.npz files, so the band here is the same statistic
# the tables quote.
#
# One figure per (sequence, modality): x = cumulative environment steps, y = mean
# return on the BASE env, one line per method, shaded band = spread across seeds.
# Modalities are never drawn on shared axes - they run different budgets and are
# not commensurable.
#
# Reading the figure:
#   - The shaded left region is task 0 (base training). No CL constraint is active
#     yet for EWC/A-GEM, so differences there are seed noise, not method effects;
#     PackNet does differ, because its prune + finetune happens inside task 0.
#   - After the first boundary the curve is an *eval* of the base env taken while
#     the agent trains on some other task. This is the forgetting signal: how fast
#     the original skill decays once training moves on.
#   - Hollow rings mark eval points where some episodes hit the eval-scan cap
#     (completed_frac < 1 in any seed); those returns are unreliable. They cluster
#     in the reward sequence, whose mods lengthen episodes.
#
# Usage:
#   python visualization/plot_curve_methods.py                       # all 8 figures
#   python visualization/plot_curve_methods.py --sequence dyn4 --modality oc
#   python visualization/plot_curve_methods.py --grid --modality pixel --smooth 5
#   python visualization/plot_curve_methods.py --band sem --seeds --format pdf
# =============================================================================

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from crl_data import (
    DEFAULT_AGG_ROOT,
    GRID,
    INK,
    INK_2,
    METHOD_COLOR,
    METHOD_LABEL,
    METHOD_ORDER,
    MODALITY_LABEL,
    MODALITY_ORDER,
    MUTED,
    SEQUENCE_LABEL,
    SEQUENCE_ORDER,
    SERIES,
    load_all,
    order_by,
    place_labels,
)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "runs" / "figures"


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered rolling mean (window <= 1 -> unchanged)."""
    if window <= 1:
        return y
    kernel = np.ones(window)
    mask = ~np.isnan(y)
    num = np.convolve(np.where(mask, y, 0.0), kernel, mode="same")
    den = np.convolve(mask.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore"):
        out = num / den
    out[~mask] = np.nan  # keep warm-up gaps as gaps rather than interpolating over them
    return out


def curve_of(agg: dict, band: str, window: int) -> dict | None:
    """(x in Msteps, mean, spread, per-seed rows, segment metadata) for one method."""
    c = agg.get("crl_curve")
    if not c:
        return None
    b = c["base_return"]
    spread = np.zeros_like(b["mean"]) if band == "none" else np.asarray(b[band], dtype=float)
    return {
        "x": np.asarray(c["env_step"], dtype=float) / 1e6,
        "mean": smooth(np.asarray(b["mean"], dtype=float), window),
        "spread": smooth(spread, window),
        "seeds": np.array([smooth(row, window) for row in np.asarray(b["seeds"], dtype=float)]),
        "task_idx": np.asarray(c["task_idx"], dtype=int),
        "task_label": np.asarray(c["task_label"]),
        "truncated": np.asarray(c["completed_frac_min"], dtype=float) < 1.0,
        "is_eval": np.asarray(c["source"]) == "eval",
        "method": agg["method"],
        "n_seeds": agg["n_seeds"],
    }


def draw_panel(ax, curves: list[dict], band: str, show_seeds: bool, annotate_tasks: bool) -> int:
    """One sequence+modality comparison onto `ax`; returns the truncated-point count."""
    ref = curves[0]
    x, task_idx = ref["x"], ref["task_idx"]
    boundaries = np.where(np.diff(task_idx) > 0)[0]

    # Task 0 is base training, not a forgetting measurement - set it apart so the
    # eye starts reading at the first boundary. Labelled in-place rather than in a
    # subtitle, which would collide with the task labels along the top.
    if boundaries.size:
        ax.axvspan(x[0], x[boundaries[0]], color=GRID, alpha=0.30, linewidth=0, zorder=0)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((x[0] + x[boundaries[0]]) / 2, 0.02, "base training", transform=trans,
                ha="center", va="bottom", color=MUTED, fontsize=7.5, style="italic", zorder=1)
    for b in boundaries:
        ax.axvline(x[b], color=GRID, linewidth=0.8, zorder=1)

    truncated_total = 0
    labels = []
    for c in curves:
        color = METHOD_COLOR.get(c["method"], SERIES[len(labels) % len(SERIES)])
        if show_seeds:
            for row in c["seeds"]:
                ax.plot(c["x"], row, color=color, linewidth=0.6, alpha=0.30, zorder=2)
        elif band != "none" and c["n_seeds"] > 1:
            ax.fill_between(c["x"], c["mean"] - c["spread"], c["mean"] + c["spread"],
                            color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(c["x"], c["mean"], color=color, linewidth=2.0, solid_capstyle="round",
                label=METHOD_LABEL.get(c["method"], c["method"]), zorder=4)

        flag = c["truncated"] & c["is_eval"]
        if flag.any():
            ax.plot(c["x"][flag], c["mean"][flag], "o", markersize=5.5, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.0, zorder=5)
            truncated_total += int(flag.sum())

        finite = np.flatnonzero(np.isfinite(c["mean"]))
        if finite.size:
            last = finite[-1]
            labels.append((c["x"][last], c["mean"][last],
                           METHOD_LABEL.get(c["method"], c["method"]), color))

    if annotate_tasks:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for t in np.unique(task_idx):
            seg = np.flatnonzero(task_idx == t)
            ax.text(x[seg].mean(), 1.015, str(ref["task_label"][seg[0]]), transform=trans,
                    ha="center", va="bottom", color=MUTED, fontsize=7)

    ax.margins(x=0.02)
    place_labels(ax, labels)
    return truncated_total


def style_axes(ax, title: str, show_xlabel=True, show_ylabel=True, pad=14) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=pad)
    if show_xlabel:
        ax.set_xlabel("Environment steps (millions)", color=INK_2, fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Return on base env", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


BAND_NOTE = {
    "std": "band = ±1 std over seeds",
    "sem": "band = ±1 standard error of the mean",
    "ci95": "band = 95% Student-t interval",
    "none": "",
}


def collect(groups: dict, sequence: str, modality: str, band: str, window: int) -> list[dict]:
    present = [a for a in groups.values()
               if a["sequence"] == sequence and a["modality"] == modality]
    by_method = {a["method"]: a for a in present}
    curves = []
    for m in order_by(by_method, METHOD_ORDER):
        c = curve_of(by_method[m], band, window)
        if c is None:
            print(f"[fig] {by_method[m]['group']}: no crl_curve data, skipping")
            continue
        curves.append(c)
    return curves


def figure_note(curves: list[dict], band: str) -> str:
    n = {c["n_seeds"] for c in curves}
    seeds = f"{n.pop()} seeds" if len(n) == 1 else "mixed seed counts"
    return ", ".join(bit for bit in (seeds, BAND_NOTE[band]) if bit)


def main() -> None:
    ap = argparse.ArgumentParser(description="Base-env return curves, all CL methods on one axes.")
    ap.add_argument("--agg-root", type=Path, default=DEFAULT_AGG_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--sequence", nargs="+", default=None, help="default: all sequences")
    ap.add_argument("--modality", nargs="+", default=MODALITY_ORDER)
    ap.add_argument("--band", choices=["std", "sem", "ci95", "none"], default="std")
    ap.add_argument("--seeds", action="store_true",
                    help="draw the individual seed traces instead of the band (honest at n=3)")
    ap.add_argument("--smooth", type=int, default=1, help="rolling-mean window in curve points")
    ap.add_argument("--grid", action="store_true",
                    help="one small-multiples figure per modality (all sequences) instead of separate files")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    groups = load_all(args.agg_root)
    sequences = order_by(args.sequence or {a["sequence"] for a in groups.values()}, SEQUENCE_ORDER)
    modalities = order_by(args.modality, MODALITY_ORDER)
    args.out.mkdir(parents=True, exist_ok=True)
    truncated_total = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN warm-up slices
        for modality in modalities:
            panels = [(s, collect(groups, s, modality, args.band, args.smooth)) for s in sequences]
            panels = [(s, c) for s, c in panels if c]
            if not panels:
                continue

            if args.grid:
                ncol = 2 if len(panels) > 1 else 1
                nrow = int(np.ceil(len(panels) / ncol))
                fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.6 * nrow),
                                         dpi=args.dpi, squeeze=False)
                for ax in axes.ravel()[len(panels):]:
                    ax.set_visible(False)
                for k, ((seq, curves), ax) in enumerate(zip(panels, axes.ravel())):
                    truncated_total += draw_panel(ax, curves, args.band, args.seeds, annotate_tasks=False)
                    style_axes(ax, SEQUENCE_LABEL.get(seq, seq),
                               show_xlabel=k >= len(panels) - ncol, show_ylabel=k % ncol == 0)
                handles, texts = axes.ravel()[0].get_legend_handles_labels()
                fig.legend(handles, texts, frameon=False, fontsize=8, labelcolor=INK_2,
                           loc="lower center", ncol=len(texts), bbox_to_anchor=(0.5, -0.01))
                fig.suptitle(f"Base-env return across task sequences — {MODALITY_LABEL[modality]}",
                             color=INK, fontsize=12, x=0.005, ha="left")
                fig.text(0.995, 0.995, figure_note(panels[0][1], args.band), ha="right", va="top",
                         color=MUTED, fontsize=8)
                fig.tight_layout(rect=(0, 0.02, 1, 0.96))
                path = args.out / f"crl_curve_grid_{modality}.{args.format}"
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"[fig] wrote {path}")
                continue

            for seq, curves in panels:
                fig, ax = plt.subplots(figsize=(9, 4.5), dpi=args.dpi)
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                truncated_total += draw_panel(ax, curves, args.band, args.seeds, annotate_tasks=True)
                style_axes(
                    ax,
                    f"Base-env return over the {SEQUENCE_LABEL.get(seq, seq).lower()} sequence"
                    f" — {MODALITY_LABEL[modality]}",
                    pad=26,  # room for the task labels running along the top
                )
                # Legend below the plot: in-plot placement collides with whichever
                # method happens to sit low, and that varies per sequence.
                ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=len(curves),
                          loc="upper left", bbox_to_anchor=(0, -0.16))
                ax.text(1.0, 1.10, figure_note(curves, args.band), transform=ax.transAxes,
                        ha="right", color=MUTED, fontsize=8)
                ax.text(0.0, -0.30, "After the shaded region the base env is only evaluated, "
                        "while the agent trains on the task named above each segment.",
                        transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=7.5)
                fig.tight_layout()
                path = args.out / f"crl_curve_{seq}_{modality}.{args.format}"
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"[fig] wrote {path}")

    if truncated_total:
        print(f"[fig] {truncated_total} eval point(s) had truncated episodes "
              f"(completed_frac < 1) - drawn as hollow rings.")


if __name__ == "__main__":
    main()
