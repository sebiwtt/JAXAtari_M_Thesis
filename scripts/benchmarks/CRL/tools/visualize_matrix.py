# =============================================================================
# Visualization tooling for CRL retention matrices
# =============================================================================
# Loads a run directory's matrix.{npz,json} (as saved by ppo_crl_continual.py) and
# renders a single multi-panel figure summarizing continual-learning performance:
#
#   1. Retention heatmap    R_norm[i,j] = (R[i,j]-R_rand[j]) / (R[j,j]-R_rand[j])
#                           1.0 = fully retained post-task-j skill, 0.0 = random floor.
#   2. Raw-return heatmap    R[i,j], with the R_rand random-agent floor as a top row.
#   3. Forgetting curves     one line per task j: retention of task j as later tasks i>=j
#                            are learned (shows *how fast* each task degrades).
#   4. Aggregate metrics     final avg performance, avg retention, backward transfer.
#
# Usage:
#   python tools/visualize_matrix.py runs/pong_ppo_crl_continual_pixel_1
#   python tools/visualize_matrix.py runs/<run> --out fig.png        # custom output path
#   python tools/visualize_matrix.py runs/<run> --show               # also open a window
# =============================================================================

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # switched to an interactive backend below if --show is set
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Okabe-Ito colorblind-safe categorical palette, assigned in fixed order (one hue per
# task, never cycled). Supports up to 8 tasks distinctly; beyond that they repeat.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def load_matrix(run_dir: str) -> dict:
    """Load matrix data from a run directory, preferring the JSON (carries metadata)."""
    json_path = os.path.join(run_dir, "matrix.json")
    npz_path = os.path.join(run_dir, "matrix.npz")

    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return {
            "R": np.array(d["R"], dtype=float),
            "R_rand": np.array(d["R_rand"], dtype=float),
            "Retention": np.array(d["Retention"], dtype=float),
            "labels": list(d["labels"]),
            "env_id": d.get("env_id", "unknown"),
            "exp_name": d.get("exp_name", "unknown"),
        }
    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return {
            "R": z["R"].astype(float),
            "R_rand": z["R_rand"].astype(float),
            "Retention": z["Retention"].astype(float),
            "labels": [str(l) for l in z["labels"]],
            # env_id/exp_name added later; fall back for older matrices.
            "env_id": str(z["env_id"]) if "env_id" in z.files else "unknown",
            "exp_name": str(z["exp_name"]) if "exp_name" in z.files else "unknown",
        }
    raise FileNotFoundError(f"No matrix.json or matrix.npz found in {run_dir!r}")


def _annotate_heatmap(ax, M, fmt="{:.2f}", threshold=None):
    """Write each cell's value; NaN cells are left blank. Text color flips for contrast."""
    vals = M[~np.isnan(M)]
    if threshold is None and vals.size:
        threshold = (np.nanmin(M) + np.nanmax(M)) / 2.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            color = "white" if (threshold is not None and M[i, j] < threshold) else "black"
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color=color, fontsize=8)


def plot_retention_heatmap(ax, Retention, labels):
    n = len(labels)
    # Diverging: red (forgot, <=0) -> yellow (half) -> green (fully retained, 1.0),
    # neutral midpoint at 0.5. Color clamps to [0,1]; annotations show true values.
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(np.clip(Retention, 0.0, 1.0), cmap="RdYlGn", norm=norm, aspect="equal")
    _annotate_heatmap(ax, Retention, threshold=0.5)

    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("evaluated on task j")
    ax.set_ylabel("trained through task i")
    ax.set_title("Retention  (1.0 = fully retained, 0.0 = random floor)")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("retention")


def plot_return_heatmap(ax, R, R_rand, labels):
    n = len(labels)
    # Stack the random-agent floor as an extra top row for context against the trained R.
    stacked = np.vstack([R_rand[None, :], R])
    finite = stacked[np.isfinite(stacked)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    im = ax.imshow(stacked, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    _annotate_heatmap(ax, stacked, fmt="{:.1f}")

    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n + 1), ["random"] + labels, fontsize=8)
    ax.set_xlabel("evaluated on task j")
    ax.set_ylabel("trained through task i")
    ax.set_title("Mean episodic return R[i, j]")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("return")


def plot_forgetting_curves(ax, Retention, labels):
    """For each task j, retention of task j at every later training stage i >= j."""
    n = len(labels)
    stages = np.arange(n)
    for j in range(n):
        color = OKABE_ITO[j % len(OKABE_ITO)]
        # Only i >= j is defined (retention is measured after task j is learned).
        ys = Retention[j:, j]
        xs = stages[j:]
        finite = np.isfinite(ys)
        if not finite.any():
            continue
        ax.plot(xs[finite], ys[finite], marker="o", markersize=5, linewidth=2,
                color=color, label=labels[j])
        # Direct label at the line's right end so identity isn't legend-only.
        ax.annotate(labels[j], (xs[finite][-1], ys[finite][-1]),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=7, color=color)

    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)  # freshly-learned
    ax.axhline(0.0, color="gray", linewidth=1, linestyle=":", alpha=0.6)   # random floor
    ax.set_xticks(stages, labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("training stage (trained through task i)")
    ax.set_ylabel("retention of task j")
    ax.set_title("Forgetting curves (per task)")
    if n <= 4:
        ax.legend(fontsize=7, loc="lower left")


def compute_metrics(R, R_rand, Retention, labels):
    """Standard continual-learning summary scalars."""
    n = len(labels)
    diag = np.diag(R)
    last_row = R[n - 1, :]  # final agent evaluated on every task

    # Average retention over the below-diagonal cells actually filled (j < i).
    lower = [Retention[i, j] for i in range(n) for j in range(i) if np.isfinite(Retention[i, j])]
    # Backward transfer: how final performance on an earlier task compares to right
    # after it was learned. Negative = forgetting.
    bwt = [R[n - 1, j] - diag[j] for j in range(n - 1) if np.isfinite(R[n - 1, j]) and np.isfinite(diag[j])]

    return {
        "final_avg_return": float(np.nanmean(last_row)),
        "final_avg_retention": float(np.nanmean(Retention[n - 1, :])),
        "avg_retention_lower": float(np.mean(lower)) if lower else float("nan"),
        "backward_transfer": float(np.mean(bwt)) if bwt else float("nan"),
    }


def plot_metrics_panel(ax, metrics):
    ax.axis("off")
    lines = [
        ("Final avg return", f"{metrics['final_avg_return']:.2f}"),
        ("Final avg retention", f"{metrics['final_avg_retention']:.3f}"),
        ("Avg retention (j<i)", f"{metrics['avg_retention_lower']:.3f}"),
        ("Backward transfer", f"{metrics['backward_transfer']:.2f}"),
    ]
    ax.set_title("Summary metrics", fontsize=11, loc="left")
    y = 0.85
    for name, val in lines:
        ax.text(0.02, y, name, fontsize=10, va="center")
        ax.text(0.98, y, val, fontsize=11, va="center", ha="right", fontweight="bold",
                family="monospace")
        y -= 0.18


def visualize(run_dir: str, out_path: str | None, show: bool) -> str:
    data = load_matrix(run_dir)
    R, R_rand, Retention = data["R"], data["R_rand"], data["Retention"]
    labels = data["labels"]
    metrics = compute_metrics(R, R_rand, Retention, labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_retention_heatmap(axes[0, 0], Retention, labels)
    plot_return_heatmap(axes[0, 1], R, R_rand, labels)
    plot_forgetting_curves(axes[1, 0], Retention, labels)
    plot_metrics_panel(axes[1, 1], metrics)

    fig.suptitle(
        f"{data['env_id']}  |  {data['exp_name']}  |  {len(labels)} tasks",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = os.path.join(run_dir, "visualization.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[viz] saved figure to {out_path}")
    print("[viz] metrics:", json.dumps(metrics, indent=2))
    if show:
        plt.show()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize a CRL retention matrix.")
    parser.add_argument("run_dir", help="run directory containing matrix.json / matrix.npz")
    parser.add_argument("--out", default=None, help="output image path (default: <run_dir>/visualization.png)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)  # noqa: needs re-import of pyplot backend
    visualize(args.run_dir, args.out, args.show)


if __name__ == "__main__":
    main()
