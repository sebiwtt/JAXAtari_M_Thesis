# =============================================================================
# Result tables from the seed-averaged CRL aggregates
# =============================================================================
# Layout (the default `main` mode): one table per metric, split into a block per
# modality; rows = CL method, columns = task sequence, cells = mean +- spread
# across seeds. The comparison being made is method-vs-method, so methods are the
# rows (a vertical scan is the easiest read) and the sequence effect falls out as
# a horizontal scan. Modality splits into blocks rather than columns because oc
# and pixel returns are not commensurable - they should never be averaged by eye.
#
# The trailing "All" column pools every seed-level value for that method and
# modality (4 sequences x 3 seeds = 12 runs) rather than averaging the four cell
# means, so it is a real 12-sample estimate. At 3 seeds per cell that column is
# where any cross-method claim has power; individual cells do not.
#
# `--per-task` switches to the drill-down: rows = method, columns = task, cells =
# Forgetting[j] (the recency-weighted drop on task j), for one sequence+modality.
# The last task has no later checkpoint, so its column is always empty.
#
# Usage:
#   python visualization/make_tables.py                                  # markdown to stdout
#   python visualization/make_tables.py --metric mean_forgetting final_avg_return
#   python visualization/make_tables.py --format latex --out tables/
#   python visualization/make_tables.py --per-task --spread sem
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crl_data import (
    DEFAULT_AGG_ROOT,
    METHOD_LABEL,
    METHOD_ORDER,
    METRIC_SPEC,
    MODALITY_LABEL,
    MODALITY_ORDER,
    SEQUENCE_LABEL,
    SEQUENCE_ORDER,
    load_all,
    order_by,
)

EMPTY = "--"
# Marks a cell backed by fewer seeds than the group nominally has (a seed whose
# metric was undefined, e.g. a collapsed base task). Never silently averaged away.
INCOMPLETE_MARK = "*"


class Cell:
    """One table entry: the numbers plus how they should be rendered."""

    def __init__(self, mean=np.nan, spread=np.nan, n=0, n_expected=0, dp=3):
        self.mean, self.spread, self.n, self.n_expected, self.dp = mean, spread, n, n_expected, dp
        self.best = False

    @property
    def empty(self) -> bool:
        return not np.isfinite(self.mean)

    @property
    def incomplete(self) -> bool:
        return 0 < self.n < self.n_expected

    def _parts(self) -> tuple[str, str]:
        return f"{self.mean:.{self.dp}f}", ("" if not np.isfinite(self.spread) else f"{self.spread:.{self.dp}f}")

    def markdown(self) -> str:
        if self.empty:
            return EMPTY
        mean, spread = self._parts()
        text = f"{mean} ± {spread}" if spread else mean
        if self.best:
            text = f"**{text}**"
        return text + (INCOMPLETE_MARK if self.incomplete else "")

    def latex(self) -> str:
        if self.empty:
            return EMPTY
        mean, spread = self._parts()
        mean = rf"\mathbf{{{mean}}}" if self.best else mean
        text = rf"${mean} \pm {spread}$" if spread else f"${mean}$"
        return text + (rf"\textsuperscript{{{INCOMPLETE_MARK}}}" if self.incomplete else "")

    def csv(self) -> str:
        if self.empty:
            return ""
        mean, spread = self._parts()
        return f"{mean} ± {spread}" if spread else mean


def spread_of(bundle: dict, kind: str) -> np.ndarray:
    return np.full_like(np.asarray(bundle["mean"], dtype=float), np.nan) if kind == "none" else bundle[kind]


def pooled_cell(bundles: list[dict], kind: str, dp: int, n_expected: int) -> Cell:
    """Pool the raw per-seed values across several groups into one cell.

    Pooling the seeds (not averaging the group means) keeps this an honest
    n-sample estimate; with equal seed counts per group the mean is identical
    either way, but the spread is not.
    """
    vals = np.concatenate([np.atleast_1d(np.asarray(b["seeds"], dtype=float).ravel()) for b in bundles])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Cell(dp=dp)
    std = np.std(vals, ddof=1) if vals.size >= 2 else np.nan
    spread = {"std": std, "sem": std / np.sqrt(vals.size), "none": np.nan}.get(
        kind, std / np.sqrt(vals.size)  # ci95 handled below
    )
    if kind == "ci95":
        from aggregate_seeds import T95
        spread = T95.get(vals.size - 1, 1.960) * std / np.sqrt(vals.size)
    return Cell(float(np.mean(vals)), float(spread), vals.size, n_expected, dp)


def mark_best(cells: list[Cell], higher_better: bool | None) -> None:
    """Flag the best cell in a column; ties are all flagged."""
    if higher_better is None:
        return
    finite = [c for c in cells if not c.empty]
    if len(finite) < 2:
        return
    target = max(c.mean for c in finite) if higher_better else min(c.mean for c in finite)
    for c in finite:
        c.best = np.isclose(c.mean, target)


def build_main_table(groups: dict, metric: str, modality: str, spread_kind: str) -> tuple[list[str], list[str], list[list[Cell]]]:
    """rows = method, cols = sequence (+ pooled 'All'); cells = metric across seeds."""
    spec = METRIC_SPEC.get(metric, {"higher_better": None, "dp": 3})
    present = {g: a for g, a in groups.items() if a["modality"] == modality}
    methods = order_by({a["method"] for a in present.values()}, METHOD_ORDER)
    sequences = order_by({a["sequence"] for a in present.values()}, SEQUENCE_ORDER)
    by_key = {(a["method"], a["sequence"]): a for a in present.values()}

    rows: list[list[Cell]] = []
    for method in methods:
        row, bundles = [], []
        for seq in sequences:
            agg = by_key.get((method, seq))
            if agg is None or metric not in agg["stats"]:
                row.append(Cell(dp=spec["dp"]))
                continue
            b = agg["stats"][metric]
            bundles.append(b)
            row.append(Cell(float(b["mean"]), float(spread_of(b, spread_kind)),
                            int(b["n"]), agg["n_seeds"], spec["dp"]))
        n_expected = sum(by_key[(method, s)]["n_seeds"] for s in sequences if (method, s) in by_key)
        row.append(pooled_cell(bundles, spread_kind, spec["dp"], n_expected) if bundles else Cell(dp=spec["dp"]))
        rows.append(row)

    for col in range(len(sequences) + 1):
        mark_best([r[col] for r in rows], spec["higher_better"])

    header = [SEQUENCE_LABEL.get(s, s) for s in sequences] + ["All"]
    return [METHOD_LABEL.get(m, m) for m in methods], header, rows


def build_per_task_table(groups: dict, sequence: str, modality: str, spread_kind: str) -> tuple[list[str], list[str], list[list[Cell]]]:
    """rows = method, cols = task; cells = per-task Forgetting[j]."""
    present = {g: a for g, a in groups.items()
               if a["modality"] == modality and a["sequence"] == sequence}
    if not present:
        return [], [], []
    methods = order_by({a["method"] for a in present.values()}, METHOD_ORDER)
    by_method = {a["method"]: a for a in present.values()}
    labels = next(iter(present.values()))["labels"]

    rows = []
    for method in methods:
        agg = by_method[method]
        b = agg["stats"]["Forgetting"]
        mean, spread, n = b["mean"], spread_of(b, spread_kind), b["n"]
        rows.append([Cell(float(mean[j]), float(spread[j]), int(n[j]), agg["n_seeds"], 3)
                     for j in range(len(labels))])
    for col in range(len(labels)):
        mark_best([r[col] for r in rows], METRIC_SPEC["mean_forgetting"]["higher_better"])
    return [METHOD_LABEL.get(m, m) for m in methods], list(labels), rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def render_markdown(blocks: list[tuple[str, list[str], list[str], list[list[Cell]]]],
                    title: str, note: str) -> str:
    out = [f"### {title}", ""]
    for block_title, row_labels, header, rows in blocks:
        if block_title:
            out += [f"**{block_title}**", ""]
        widths = [max(len(h), *(len(r[i].markdown()) for r in rows)) for i, h in enumerate(header)]
        lw = max(len(l) for l in row_labels + [""])
        out.append("| " + " | ".join([" " * lw] + [h.ljust(w) for h, w in zip(header, widths)]) + " |")
        out.append("|" + "|".join(["-" * (lw + 2)] + ["-" * (w + 2) for w in widths]) + "|")
        for label, row in zip(row_labels, rows):
            out.append("| " + " | ".join([label.ljust(lw)] + [c.markdown().ljust(w) for c, w in zip(row, widths)]) + " |")
        out.append("")
    if note:
        out += [note, ""]
    return "\n".join(out)


def render_latex(blocks: list[tuple[str, list[str], list[str], list[list[Cell]]]],
                 title: str, note: str) -> str:
    header = blocks[0][2]
    ncol = len(header) + 1
    out = [
        r"\begin{table}[t]", r"  \centering",
        rf"  \caption{{{title}}}",
        r"  \begin{tabular}{l" + "r" * (ncol - 1) + "}", r"    \toprule",
        "    " + " & ".join([""] + header) + r" \\",
    ]
    for bi, (block_title, row_labels, _, rows) in enumerate(blocks):
        out.append(r"    \midrule")
        if block_title:
            out.append(rf"    \multicolumn{{{ncol}}}{{l}}{{\emph{{{block_title}}}}} \\")
        for label, row in zip(row_labels, rows):
            out.append("    " + " & ".join([label] + [c.latex() for c in row]) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        # \par keeps the note inside the table float, below the tabular.
        out.append(rf"  \par\smallskip\footnotesize {note}")
    out += [r"\end{table}", ""]
    return "\n".join(out)


def render_csv(blocks: list[tuple[str, list[str], list[str], list[list[Cell]]]],
               title: str, note: str) -> str:
    out = []
    for block_title, row_labels, header, rows in blocks:
        out.append(",".join([f'"{block_title or title}"'] + [f'"{h}"' for h in header]))
        for label, row in zip(row_labels, rows):
            out.append(",".join([f'"{label}"'] + [f'"{c.csv()}"' for c in row]))
        out.append("")
    return "\n".join(out)


RENDERERS = {"markdown": render_markdown, "latex": render_latex, "csv": render_csv}
SUFFIX = {"markdown": ".md", "latex": ".tex", "csv": ".csv"}


def spread_note(kind: str, blocks, fmt: str = "markdown") -> str:
    pm = r"$\pm$" if fmt == "latex" else "±"
    pct = r"95\%" if fmt == "latex" else "95%"
    bits = []
    if kind != "none":
        bits.append({"std": f"mean {pm} std over seeds",
                     "sem": f"mean {pm} standard error of the mean",
                     "ci95": f"mean {pm} half-width of the {pct} Student-t interval"}[kind])
    if any(c.incomplete for _, _, _, rows in blocks for r in rows for c in r):
        bits.append(f"{INCOMPLETE_MARK} backed by fewer seeds than the rest (metric undefined for a seed)")
    return "; ".join(bits) + ("." if bits else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Result tables from the seed-averaged CRL aggregates.")
    ap.add_argument("--agg-root", type=Path, default=DEFAULT_AGG_ROOT,
                    help=f"directory of aggregate dirs (default: {DEFAULT_AGG_ROOT})")
    ap.add_argument("--metric", nargs="+", default=["mean_forgetting"],
                    choices=sorted(METRIC_SPEC), help="one table per metric (default: mean_forgetting)")
    ap.add_argument("--modality", nargs="+", default=MODALITY_ORDER, help="oc / pixel (default: both)")
    ap.add_argument("--spread", choices=["std", "sem", "ci95", "none"], default="std",
                    help="what follows the ± (default: std)")
    ap.add_argument("--format", choices=sorted(RENDERERS), default="markdown")
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of stdout; a directory gets one file per table")
    ap.add_argument("--per-task", action="store_true",
                    help="per-task Forgetting[j] drill-down instead of the aggregate metric table")
    ap.add_argument("--sequence", nargs="+", default=None,
                    help="per-task mode: which sequences (default: all)")
    args = ap.parse_args()

    groups = load_all(args.agg_root)
    modalities = [m for m in order_by(args.modality, MODALITY_ORDER)]
    render = RENDERERS[args.format]
    tables: list[tuple[str, str]] = []  # (slug, rendered text)

    if args.per_task:
        sequences = order_by(args.sequence or {a["sequence"] for a in groups.values()}, SEQUENCE_ORDER)
        for seq in sequences:
            blocks = []
            for mod in modalities:
                row_labels, header, rows = build_per_task_table(groups, seq, mod, args.spread)
                if rows:
                    blocks.append((MODALITY_LABEL.get(mod, mod), row_labels, header, rows))
            if not blocks:
                continue
            title = f"Per-task forgetting - {SEQUENCE_LABEL.get(seq, seq)} sequence"
            tables.append((f"forgetting_per_task_{seq}",
                           render(blocks, title, spread_note(args.spread, blocks, args.format))))
    else:
        for metric in args.metric:
            blocks = []
            for mod in modalities:
                row_labels, header, rows = build_main_table(groups, metric, mod, args.spread)
                if rows:
                    blocks.append((MODALITY_LABEL.get(mod, mod), row_labels, header, rows))
            if not blocks:
                continue
            title = METRIC_SPEC[metric]["label"]
            note = spread_note(args.spread, blocks, args.format)
            quoted = "``All''" if args.format == "latex" else '"All"'
            note = (note + " " if note else "") + f"{quoted} pools every seed of all sequences."
            tables.append((metric, render(blocks, title, note)))

    if args.out is None:
        print("\n".join(text for _, text in tables))
        return
    if len(tables) > 1 or args.out.is_dir() or args.out.suffix == "":
        args.out.mkdir(parents=True, exist_ok=True)
        for slug, text in tables:
            path = args.out / f"{slug}{SUFFIX[args.format]}"
            path.write_text(text)
            print(f"[tables] wrote {path}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tables[0][1])
        print(f"[tables] wrote {args.out}")


if __name__ == "__main__":
    main()
