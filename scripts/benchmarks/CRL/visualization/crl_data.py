# =============================================================================
# Shared loading / naming for the CRL visualization suite
# =============================================================================
# Every plot and table script reads the seed-averaged files written by
# aggregate_seeds.py, so the loading, the group-name parsing and the fixed
# display order of methods/sequences live here rather than being re-derived
# (and quietly diverging) in each script.
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGG_ROOT = SCRIPT_DIR.parent / "runs" / "aggregated"

# Fixed display order. Methods: the naive baseline first, then the CL methods in
# increasing structural intervention (regularization -> gradient projection ->
# parameter isolation). Sequences: the perturbation axes in the thesis' order.
METHOD_ORDER = ["ft", "ewc", "agem", "packnet"]
SEQUENCE_ORDER = ["dyn4", "vis4", "rew4", "mag4"]
MODALITY_ORDER = ["oc", "pixel"]

METHOD_LABEL = {"ft": "Fine-tuning", "ewc": "EWC", "agem": "A-GEM", "packnet": "PackNet"}

# Categorical palette, shared with tools/plot_crl_curve.py so a method keeps its
# color across every figure in the thesis ("color follows the entity"). Validated
# as a 4-slot adjacent-pair set on a light surface: worst adjacent CVD dE 9.1
# (protan), worst normal-vision dE 22.9. Aqua and yellow fall under 3:1 contrast
# against white, so any figure using them owes visible direct labels - which the
# curve plot ships.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLOR = {"ft": SERIES[0], "ewc": SERIES[1], "agem": SERIES[2], "packnet": SERIES[3]}
# Text/structure tokens: never use a series color for text.
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Sequential ramp (one hue, light -> dark) for MAGNITUDE scales - retention and drop
# are "how much of the skill is left", not a polarity around a neutral, so they take
# this rather than a diverging map. (The pre-aggregate tools/visualize_matrix.py used
# RdYlGn, which puts a hue at the midpoint and is red-green: unreadable under the two
# most common CVD types.)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Diverging arms for SIGNED scales (raw return: losing < 0 < winning), gray midpoint.
RED_RAMP = ["#f7d6d6", "#f0b4b4", "#ea9191", "#e66d6d", "#e34948", "#c73a39", "#a82e2d",
            "#8a2322", "#6d1a19"]
NEUTRAL = "#f0efec"


def sequential_cmap(name: str = "crl_seq"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, BLUE_RAMP)


def diverging_cmap(name: str = "crl_div"):
    """Blue (low) -> neutral gray -> red (high); equal steps per arm."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, BLUE_RAMP[::-1] + [NEUTRAL] + RED_RAMP)


SEQUENCE_LABEL = {"dyn4": "Dynamics", "vis4": "Visual", "rew4": "Reward", "mag4": "Magnitude"}
MODALITY_LABEL = {"oc": "Object-centric", "pixel": "Pixel"}

# runs/aggregated/<env>_<method>_<sequence>_<modality>
GROUP_RE = re.compile(r"^(?P<env>[^_]+)_(?P<method>[^_]+)_(?P<sequence>[^_]+)_(?P<modality>oc|pixel)$")

# How each metric should be read and printed. `higher_better` drives which cell a
# table marks as best (None = no direction, e.g. wall-clock).
METRIC_SPEC = {
    "mean_forgetting":       {"label": "Mean forgetting",        "higher_better": False, "dp": 3},
    "mean_retention":        {"label": "Mean retention",         "higher_better": True,  "dp": 3},
    "final_avg_return":      {"label": "Final avg return",       "higher_better": True,  "dp": 2},
    "final_avg_retention":   {"label": "Final avg retention",    "higher_better": True,  "dp": 3},
    "avg_retention_lower":   {"label": "Avg retention (j<i)",    "higher_better": True,  "dp": 3},
    "avg_final_return_norm": {"label": "Final return (norm.)",   "higher_better": True,  "dp": 3},
    "backward_transfer":     {"label": "Backward transfer",      "higher_better": True,  "dp": 2},
    "total_compute_time_sec": {"label": "Compute time (s)",      "higher_better": None,  "dp": 0},
}


def parse_group(group: str) -> dict[str, str]:
    m = GROUP_RE.match(group)
    if m is None:
        raise ValueError(f"cannot parse group name {group!r} as <env>_<method>_<sequence>_<modality>")
    return m.groupdict()


def load_aggregate(path: str | Path) -> dict:
    """Read one aggregate.json (or the directory containing it) into numpy arrays."""
    path = Path(path)
    if path.is_dir():
        path = path / "aggregate.json"
    with open(path) as f:
        agg = json.load(f)
    for bundle in agg["stats"].values():
        for stat, val in bundle.items():
            bundle[stat] = np.asarray(val, dtype=int if stat == "n" else float)
    if agg.get("crl_curve"):
        c = agg["crl_curve"]
        for k, v in c.items():
            if k == "base_return":
                for stat, val in v.items():
                    v[stat] = np.asarray(val, dtype=int if stat == "n" else float)
            else:
                c[k] = np.asarray(v)
    agg.update(parse_group(agg["group"]))
    return agg


def load_all(agg_root: str | Path = DEFAULT_AGG_ROOT) -> dict[str, dict]:
    """All aggregates under `agg_root`, keyed by group name."""
    agg_root = Path(agg_root)
    if not agg_root.is_dir():
        raise SystemExit(
            f"{agg_root} not found - run visualization/aggregate_seeds.py first"
        )
    out = {}
    for d in sorted(agg_root.iterdir()):
        if (d / "aggregate.json").exists():
            agg = load_aggregate(d)
            out[agg["group"]] = agg
    if not out:
        raise SystemExit(f"no aggregate.json found under {agg_root}")
    return out


def order_by(values: set[str] | list[str], preferred: list[str]) -> list[str]:
    """Preferred order first, then anything unrecognized, alphabetically."""
    seen = list(dict.fromkeys(values))
    known = [v for v in preferred if v in seen]
    return known + sorted(v for v in seen if v not in preferred)


def place_labels(ax, entries, min_gap_frac: float = 0.055) -> None:
    """Direct labels at each series' right end, nudged apart so they never collide.

    `entries` is [(x, y, text, color)]. Required, not decorative: several palette
    slots sit under 3:1 contrast on white, so identity must not rest on color
    alone - and series that converge (or are all identically zero) would otherwise
    stack their labels into an unreadable pile.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    entries = sorted(entries, key=lambda e: e[1])
    gap = min(min_gap_frac * span, span / max(len(entries), 1))
    ys = [e[1] for e in entries]
    for _ in range(200):
        moved = False
        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] < gap:
                shift = (gap - (ys[i + 1] - ys[i])) / 2
                ys[i] -= shift
                ys[i + 1] += shift
                moved = True
        if not moved:
            break
    # Slide the block back inside the axes; a label pushed past the top is
    # otherwise cropped out of the saved figure without warning.
    overflow = max(0.0, ys[-1] - (hi - 0.01 * span)) or min(0.0, ys[0] - (lo + 0.01 * span))
    ys = [y - overflow for y in ys]
    for (x, y0, text, color), y in zip(entries, ys):
        ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=color, clip_on=False)
        if abs(y - y0) > 0.02 * span:
            ax.plot([x, x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])], [y0, y],
                    color=color, linewidth=0.6, alpha=0.55, clip_on=False, zorder=3)
