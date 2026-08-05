# =============================================================================
# Retention / forgetting matrix dashboard for a seed-averaged group
# =============================================================================
# The aggregate-fed analog of tools/visualize_matrix.py. Same six panels, but
# every number is a mean over seed replicates and every panel shows the spread
# rather than hiding it:
#
#   1. Retention heatmap      mean, with +-std under each cell
#   2. Drop heatmap           mean, with +-std under each cell
#   3. Raw-return heatmap     mean R[i,j] (+ the random-agent floor row)
#   4. Forgetting curves      per-task drop over training stages, +-std band
#   5. Summary metrics        mean +- std for each CL scalar
#   6. Per-task drop bars     bars = mean, whiskers = std, dots = the seeds
#
# Two deliberate departures from the single-run tool:
#
#   Color. Retention and drop are MAGNITUDES ("how much of the skill is left"),
#   so they take a one-hue sequential ramp. The old RdYlGn is diverging, puts a
#   hue at its midpoint, and is red-green - the two commonest CVD types cannot
#   read it. Raw return IS signed (losing < 0 < winning in Pong), so it keeps a
#   diverging map, centered on 0 with a neutral gray midpoint.
#
#   Cell counts. A cell whose metric was undefined for some seed (e.g. a seed
#   whose base task collapsed, leaving Retention[:, 0] undefined) is averaged
#   over fewer seeds; those cells are marked "n=k" rather than passing for a
#   full-strength average.
#
# Usage:
#   python visualization/plot_matrix.py                          # every group
#   python visualization/plot_matrix.py pong_ft_dyn4_oc
#   python visualization/plot_matrix.py --modality pixel --format pdf
#   python visualization/plot_matrix.py pong_ft_dyn4_oc --bars   # panel 6 alone
# =============================================================================

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from crl_data import (
    BLUE_RAMP,
    DEFAULT_AGG_ROOT,
    GRID,
    INK,
    INK_2,
    METHOD_LABEL,
    MODALITY_LABEL,
    MUTED,
    SEQUENCE_LABEL,
    SERIES,
    diverging_cmap,
    load_all,
    place_labels,
    sequential_cmap,
)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "runs" / "figures"
SEQ = sequential_cmap()
DIV = diverging_cmap()


def _stat(agg: dict, name: str, stat: str) -> np.ndarray:
    return np.asarray(agg["stats"][name][stat], dtype=float)


def _ink_on(cmap, norm, value: float) -> str:
    """Readable ink for a cell, from the luminance of the color actually painted.

    A normalized-value threshold would be wrong for the diverging map, whose BOTH
    ends are dark; asking the colormap what it produced works for either scale.
    """
    r, g, b, _ = cmap(norm(value))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if luminance < 0.35 else INK


def annotate_cells(ax, mean, spread, n, n_seeds, norm, cmap, fmt="{:.2f}"):
    """mean on top, ±std beneath, and n where a cell lost seeds. NaN cells stay blank."""
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if not np.isfinite(mean[i, j]):
                continue
            color = _ink_on(cmap, norm, mean[i, j])
            ax.text(j, i - 0.13, fmt.format(mean[i, j]), ha="center", va="center",
                    color=color, fontsize=8.5)
            if np.isfinite(spread[i, j]):
                ax.text(j, i + 0.17, f"±{spread[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=6.5, alpha=0.85)
            if 0 < n[i, j] < n_seeds:
                ax.text(j + 0.46, i - 0.42, f"n={int(n[i, j])}", ha="right", va="top",
                        color=color, fontsize=5.5, style="italic")


def _axis_labels(ax, labels, ylabels=None):
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8, color=INK_2)
    yl = labels if ylabels is None else ylabels
    ax.set_yticks(range(len(yl)), yl, fontsize=8, color=INK_2)
    ax.set_xlabel("evaluated on task j", fontsize=9, color=INK_2)
    ax.set_ylabel("trained through task i", fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_unit_heatmap(ax, agg, name: str, title: str, cbar_label: str):
    """Retention or Drop: both live in [0,1], so they share one fixed scale."""
    mean, spread, n = (_stat(agg, name, s) for s in ("mean", "std", "n"))
    norm = Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(mean, cmap=SEQ, norm=norm, aspect="equal")
    annotate_cells(ax, mean, spread, n, agg["n_seeds"], norm, SEQ)
    _axis_labels(ax, agg["labels"])
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=8)


def plot_return_heatmap(ax, agg):
    """R[i,j] with the random-agent floor stacked on top as a reference row."""
    mean, spread, n = (_stat(agg, "R", s) for s in ("mean", "std", "n"))
    rand_m, rand_s, rand_n = (_stat(agg, "R_rand", s) for s in ("mean", "std", "n"))
    stacked = np.vstack([rand_m[None, :], mean])
    stacked_s = np.vstack([rand_s[None, :], spread])
    stacked_n = np.vstack([rand_n[None, :], n])

    finite = stacked[np.isfinite(stacked)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
    # Center on 0: negative return means losing the rally, positive means winning.
    # TwoSlopeNorm needs a strictly increasing triple, so pad a degenerate side.
    norm = TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=max(hi, 1e-6))
    im = ax.imshow(stacked, cmap=DIV, norm=norm, aspect="equal")
    annotate_cells(ax, stacked, stacked_s, stacked_n, agg["n_seeds"], norm, DIV, fmt="{:.1f}")
    _axis_labels(ax, agg["labels"], ylabels=["random"] + list(agg["labels"]))
    ax.set_title("Mean episodic return R[i, j]", fontsize=10, color=INK, loc="left")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("return", fontsize=8)


def plot_forgetting_curves(ax, agg, band: bool = True):
    """For each task j, the drop on task j at every later stage i > j (origin 0 by definition)."""
    labels = agg["labels"]
    n = len(labels)
    mean, spread = _stat(agg, "Drop", "mean"), _stat(agg, "Drop", "std")
    stages = np.arange(n)
    ends: list[tuple[float, float, str, str]] = []
    for j in range(n - 1):  # the last task has no later stage
        color = SERIES[j % len(SERIES)]
        # Drop[j, j] is not stored (trivially 0: nothing later to compare against yet);
        # prepend it so the curve starts at its true origin.
        ys = np.concatenate([[0.0], mean[j + 1:, j]])
        es = np.concatenate([[0.0], spread[j + 1:, j]])
        xs = stages[j:]
        ok = np.isfinite(ys)
        if not ok.any():
            continue
        if band:
            # Drop is bounded [0,1] by construction, so a symmetric ±std band would
            # draw into regions the quantity cannot occupy.
            ax.fill_between(xs[ok], np.clip(ys - es, 0, 1)[ok], np.clip(ys + es, 0, 1)[ok],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(xs[ok], ys[ok], marker="o", markersize=4.5, linewidth=2, color=color, label=labels[j])
        ends.append((xs[ok][-1], ys[ok][-1], labels[j], color))

    ax.axhline(0.0, color=GRID, linewidth=1, linestyle="--")  # no forgetting
    # Drop's own domain, fixed: keeps panels comparable across methods, and stops a
    # method with zero forgetting everywhere from autoscaling to a meaningless ±0.04.
    ax.set_ylim(-0.03, 1.03)
    place_labels(ax, ends)
    ax.set_xticks(stages, labels, rotation=45, ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("training stage (trained through task i)", fontsize=9, color=INK_2)
    ax.set_ylabel("drop on task j", fontsize=9, color=INK_2)
    ax.set_title("Forgetting curves (per task)", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.margins(x=0.12)


METRIC_ROWS = [
    ("mean_forgetting", "Mean forgetting", 3),
    ("mean_retention", "Mean retention", 3),
    ("avg_retention_lower", "Avg retention (j<i)", 3),
    ("final_avg_retention", "Final avg retention", 3),
    ("avg_final_return_norm", "Final return (norm.)", 3),
    ("final_avg_return", "Final avg return", 2),
    ("backward_transfer", "Backward transfer", 2),
]


def plot_metrics_panel(ax, agg):
    ax.axis("off")
    ax.set_title("Summary metrics", fontsize=10, color=INK, loc="left")
    ax.text(0.98, 0.95, f"{agg['n_seeds']} seeds, ± = std", ha="right", va="center",
            fontsize=7.5, color=MUTED, transform=ax.transAxes)
    y = 0.82
    for key, label, dp in METRIC_ROWS:
        b = agg["stats"].get(key)
        if b is None:
            continue
        mean, std = float(np.asarray(b["mean"])), float(np.asarray(b["std"]))
        ax.text(0.02, y, label, fontsize=9.5, va="center", color=INK_2)
        ax.text(0.98, y, f"{mean:.{dp}f}", fontsize=11, va="center", ha="right",
                fontweight="bold", family="monospace", color=INK)
        ax.text(0.98, y - 0.055, f"± {std:.{dp}f}" if np.isfinite(std) else "",
                fontsize=7.5, va="center", ha="right", family="monospace", color=MUTED)
        y -= 0.125


def plot_drop_bars(ax, agg, show_seeds: bool = True):
    """Per-task drop, grouped by the task whose skill is being measured.

    One group per task j; within it one bar per later stage i > j. Stages are
    ORDERED, so they take a one-hue light->dark ramp rather than categorical hues.
    Whiskers are ±1 std across seeds and the dots are the seeds themselves - at
    n=3 the individual runs carry more information than any interval.
    """
    labels = agg["labels"]
    n = len(labels)
    mean, spread = _stat(agg, "Drop", "mean"), _stat(agg, "Drop", "std")
    seeds = np.asarray(agg["stats"]["Drop"]["seeds"], dtype=float)
    forg_m, forg_s = _stat(agg, "Forgetting", "mean"), _stat(agg, "Forgetting", "std")
    mean_forg = float(np.asarray(agg["stats"]["mean_forgetting"]["mean"]))

    max_bars = max(1, n - 1)
    width = 0.8 / max_bars
    ramp = [BLUE_RAMP[3], BLUE_RAMP[6], BLUE_RAMP[8], BLUE_RAMP[10], BLUE_RAMP[12]]
    stage_color = {i: ramp[min(i - 1, len(ramp) - 1)] for i in range(1, n)}

    tick_pos = []
    for j in range(n):
        stages = [i for i in range(j + 1, n) if np.isfinite(mean[i, j])]
        if not stages:
            tick_pos.append(j - 0.4 + 0.5 * width)
            # Mid-height, not at y=0: the mean-forgetting line label also lives down
            # there, and for a zero-forgetting method the two would overlap.
            ax.text(j - 0.4 + 0.5 * width, 0.5, "n/a\n(last task)", ha="center", va="center",
                    fontsize=7, color=MUTED, style="italic")
            continue
        x_start = j - 0.4
        x_end = x_start + len(stages) * width
        tick_pos.append((x_start + x_end) / 2)
        for k, i in enumerate(stages):
            x = x_start + (k + 0.5) * width
            ax.bar(x, mean[i, j], width=width * 0.88, color=stage_color[i],
                   edgecolor="white", linewidth=0.6, zorder=3)
            if np.isfinite(spread[i, j]):
                # Clipped to [0,1] for the same reason as the curve bands: a whisker
                # past either bound would claim an impossible drop.
                lo = mean[i, j] - max(0.0, mean[i, j] - spread[i, j])
                hi = min(1.0, mean[i, j] + spread[i, j]) - mean[i, j]
                ax.errorbar(x, mean[i, j], yerr=[[lo], [hi]], fmt="none", ecolor=INK_2,
                            elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
            if show_seeds:
                pts = seeds[:, i, j]
                pts = pts[np.isfinite(pts)]
                # Deterministic spread across the bar so overlapping seeds stay countable.
                offs = np.linspace(-0.22, 0.22, len(pts)) * width if len(pts) > 1 else np.zeros(1)
                ax.plot(x + offs, pts, "o", markersize=2.6, markerfacecolor="white",
                        markeredgecolor=INK_2, markeredgewidth=0.6, linestyle="none", zorder=5)
        if np.isfinite(forg_m[j]):
            ax.plot([x_start, x_end], [forg_m[j]] * 2, color=INK, linewidth=1.8,
                    solid_capstyle="butt", zorder=6)

    if np.isfinite(mean_forg):
        ax.axhline(mean_forg, color=MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.text(n - 0.45, min(mean_forg + 0.02, 1.03), f"mean {mean_forg:.3f}", ha="right",
                va="bottom", fontsize=7, color=MUTED)

    handles = [Patch(facecolor=stage_color[i], edgecolor="white", label=f"after {labels[i]}")
               for i in range(1, n)]
    handles.append(Line2D([0], [0], color=INK, lw=1.8, label="task forgetting (recency-weighted)"))
    if show_seeds:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                              markeredgecolor=INK_2, markersize=4, label="individual seeds"))
    # Outside, to the right: bars span the full width and reach 1.0, so every in-plot
    # corner collides, and below the axes runs into the rotated tick labels.
    ax.legend(handles=handles, fontsize=6.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.set_xticks(tick_pos, labels, rotation=45, ha="right", fontsize=8, color=INK_2)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("task whose performance is measured", fontsize=9, color=INK_2)
    ax.set_ylabel("performance drop", fontsize=9, color=INK_2)
    ax.set_title("Performance drop per task, by training stage\n(0 = no forgetting, 1 = fully forgotten)",
                 fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def group_title(agg: dict) -> str:
    return (f"{agg['env_id']} · {METHOD_LABEL.get(agg['method'], agg['method'])} · "
            f"{SEQUENCE_LABEL.get(agg['sequence'], agg['sequence'])} sequence · "
            f"{MODALITY_LABEL.get(agg['modality'], agg['modality'])}")


def visualize(agg: dict, out_dir: Path, fmt: str, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=dpi)
    fig.patch.set_facecolor("white")
    plot_unit_heatmap(axes[0, 0], agg, "Retention",
                      "Retention  (1.0 = fully retained, 0.0 = at/below random)", "retention")
    plot_unit_heatmap(axes[0, 1], agg, "Drop",
                      "Drop  (0.0 = no forgetting, 1.0 = fully forgotten)", "drop")
    plot_return_heatmap(axes[0, 2], agg)
    plot_forgetting_curves(axes[1, 0], agg)
    plot_metrics_panel(axes[1, 1], agg)
    plot_drop_bars(axes[1, 2], agg)

    fig.suptitle(group_title(agg), fontsize=14, fontweight="bold", color=INK)
    fig.text(0.995, 0.975, f"{agg['n_seeds']} seeds (#{', #'.join(map(str, agg['seeds']))}) · "
             f"cell values are means, ± is std across seeds",
             ha="right", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = out_dir / f"matrix_{agg['group']}.{fmt}"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def visualize_bars(agg: dict, out_dir: Path, fmt: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.patch.set_facecolor("white")
    plot_drop_bars(ax, agg)
    fig.suptitle(group_title(agg), fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"drop_bars_{agg['group']}.{fmt}"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Retention/forgetting dashboard for seed-averaged CRL groups.")
    ap.add_argument("groups", nargs="*", help="group names (default: every aggregate)")
    ap.add_argument("--agg-root", type=Path, default=DEFAULT_AGG_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--modality", nargs="+", default=None, help="filter: oc / pixel")
    ap.add_argument("--method", nargs="+", default=None, help="filter: ft / ewc / agem / packnet")
    ap.add_argument("--sequence", nargs="+", default=None, help="filter: dyn4 / vis4 / rew4 / mag4")
    ap.add_argument("--bars", action="store_true", help="also save the drop bar chart standalone")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    all_groups = load_all(args.agg_root)
    if args.groups:
        missing = [g for g in args.groups if g not in all_groups]
        if missing:
            raise SystemExit(f"unknown group(s): {', '.join(missing)}")
        selected = [all_groups[g] for g in args.groups]
    else:
        selected = list(all_groups.values())
    for key, wanted in (("modality", args.modality), ("method", args.method), ("sequence", args.sequence)):
        if wanted:
            selected = [a for a in selected if a[key] in wanted]
    if not selected:
        raise SystemExit("no groups matched the filters")

    args.out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for agg in selected:
            print(f"[viz] wrote {visualize(agg, args.out, args.format, args.dpi)}")
            if args.bars:
                print(f"[viz] wrote {visualize_bars(agg, args.out, args.format, args.dpi)}")


if __name__ == "__main__":
    main()
