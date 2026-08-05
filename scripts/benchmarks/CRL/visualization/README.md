# CRL visualization suite

Tools for turning the per-run output of `ppo_crl_continual.py` into thesis-ready
tables and figures. Everything here works on **seed-averaged groups**, not single
runs — the single-run equivalents live in `../tools/` (`visualize_matrix.py`,
`plot_crl_curve.py`) and are still the right thing for inspecting one run.

All commands below are run from `scripts/benchmarks/CRL/`.

## Pipeline

```
runs/<group>_<seed>/matrix.{json,npz}      written by ppo_crl_continual.py
        │
        │  aggregate_seeds.py              ← run this first, after any new runs
        ▼
runs/aggregated/<group>/aggregate.{json,npz}
        │
        ├── make_tables.py          → markdown / LaTeX / CSV tables
        ├── plot_curve_methods.py   → base-env return curves, methods compared
        └── plot_matrix.py          → retention/forgetting dashboard per group
```

A **group** is one configuration across its seed replicates:
`pong_ft_dyn4_oc_{1,2,3}` → group `pong_ft_dyn4_oc`. Names parse as
`<env>_<method>_<sequence>_<modality>`.

`crl_data.py` is the shared library (loading, group parsing, fixed method/sequence
ordering, the color palette). It is imported by the other scripts, not run directly.

---

## 1. `aggregate_seeds.py` — collapse seed replicates

```bash
python visualization/aggregate_seeds.py                       # every group under runs/
python visualization/aggregate_seeds.py runs/pong_ft_dyn4_oc_*
python visualization/aggregate_seeds.py runs/pong_packnet_mag4_oc_{2,3}   # drop a bad seed
```

Writes `runs/aggregated/<group>/aggregate.json` (canonical) and `.npz` (same data,
flat array keys, faster to load).

| flag | meaning |
|---|---|
| `--runs-root` | where the run dirs live (default `runs/`) |
| `--out-root` | output root (default `<runs-root>/aggregated`) |
| `--min-seeds N` | skip groups with fewer replicates (default 2) |
| `--quiet` | suppress the per-group summary lines |

**Re-run this after every new batch of runs.** Everything downstream reads only the
aggregates, so a stale `runs/aggregated/` silently produces stale figures.

### What's in an aggregate

Every quantity — `R`, `R_rand`, `Retention`, `Drop`, `Forgetting`,
`mean_forgetting`, `mean_retention`, the derived CL scalars, and the full CRL
curve — is reduced across seeds per element into:

| key | meaning |
|---|---|
| `mean` | across-seed mean (NaN-aware) |
| `std` | **sample** std, `ddof=1` |
| `sem` | `std / sqrt(n)` — uncertainty of the *mean* |
| `ci95` | half-width of the Student-t 95% interval (`t=4.303` at n=3) |
| `median`, `min`, `max` | order statistics |
| `n` | per-element count of non-NaN seeds — **not always 3** |
| `seeds` | the raw per-seed values, kept so nothing is locked in |

Two choices worth knowing:

- **Derived metrics are aggregated per seed, never recomputed from the mean `R`.**
  Retention is a clipped ratio, so mean-of-ratios ≠ ratio-of-means, and the seed is
  the sampling unit the error bars are meant to describe.
- **`std` vs `sem` vs `ci95` answer different questions.** `std` = "how much do
  seeds vary" (curve bands). `sem` = "how well do I know the mean" (error bars for
  method-vs-method claims). Don't mix them within one figure.

At n=3, skip bootstrap CIs, IQM, and significance tests — they need ~10+ runs. The
suite's approach instead is interval estimates plus showing the individual runs
(`--seeds` on the curve plot, seed dots in the matrix bar panel).

---

## 2. `make_tables.py` — result tables

```bash
python visualization/make_tables.py                                   # markdown to stdout
python visualization/make_tables.py --metric mean_forgetting avg_final_return_norm
python visualization/make_tables.py --format latex --out tables/      # one .tex per metric
python visualization/make_tables.py --per-task --sequence dyn4 --spread sem
```

Layout: **rows = method, columns = sequence, one block per modality.** Method
comparison is the question being asked, and a vertical scan is the easiest read.
Modalities are blocks rather than columns because oc and pixel returns are not
commensurable — they should never be averaged by eye.

The trailing **`All`** column pools every seed-level value for that method and
modality (4 sequences × 3 seeds = 12 runs) rather than averaging the four cell
means, so it is a real 12-sample estimate. That column is where a cross-method
claim has power; individual cells do not.

| flag | meaning |
|---|---|
| `--metric` | one table per metric; see the list below (default `mean_forgetting`) |
| `--per-task` | drill-down: rows = method, columns = task, cells = `Forgetting[j]` |
| `--modality` / `--sequence` | filters |
| `--spread {std,sem,ci95,none}` | what follows the ± (default `std`) |
| `--format {markdown,latex,csv}` | LaTeX emits booktabs; the best cell is `\mathbf` on the mean |
| `--out` | a file, or a directory to get one file per metric |

Available metrics: `mean_forgetting`, `mean_retention`, `avg_retention_lower`,
`final_avg_retention`, `avg_final_return_norm`, `final_avg_return`,
`backward_transfer`, `total_compute_time_sec`.

A `*` on a cell means it is backed by fewer seeds than the rest; the footnote
explains it. The best cell per column is bolded, using each metric's known
direction (lower is better for forgetting, higher for retention/return).

---

## 3. `plot_curve_methods.py` — base-env return over the sequence

```bash
python visualization/plot_curve_methods.py --smooth 3            # all 8 figures
python visualization/plot_curve_methods.py --sequence dyn4 --modality oc
python visualization/plot_curve_methods.py --grid --smooth 5     # 2x2 small multiples
python visualization/plot_curve_methods.py --seeds --sequence vis4 --modality oc
```

One figure per (sequence, modality): x = cumulative environment steps, y = mean
return on the **base** env, one line per method, shaded band = spread across seeds.
Output goes to `runs/figures/crl_curve_<sequence>_<modality>.<fmt>`.

| flag | meaning |
|---|---|
| `--grid` | one small-multiples figure per modality instead of separate files |
| `--seeds` | draw the individual seed traces instead of the band |
| `--band {std,sem,ci95,none}` | band statistic (default `std`) |
| `--smooth N` | rolling-mean window in curve points; `3`–`5` is usually enough |
| `--sequence` / `--modality` | filters |
| `--format {png,pdf,svg}`, `--dpi`, `--out` | output control |

### Reading the figure

- The **shaded left region is task 0** (base training). No CL constraint is active
  yet for EWC/A-GEM, so differences there are seed noise, not method effects.
  PackNet *does* differ — its prune + finetune happens inside task 0, which is the
  sharp one-point crater at 50% of the base budget.
- **After the first boundary the base env is only evaluated**, while the agent
  trains on the task named above each segment. That stretch is the forgetting
  signal.
- **Hollow rings** mark eval points where some episodes hit the eval-scan cap
  (`completed_frac < 1` in any seed) — see the caveats below.

---

## 4. `plot_matrix.py` — retention / forgetting dashboard

```bash
python visualization/plot_matrix.py                        # every group (32 figures)
python visualization/plot_matrix.py pong_ft_dyn4_oc
python visualization/plot_matrix.py --method packnet --modality oc
python visualization/plot_matrix.py pong_ft_dyn4_oc --bars --format pdf
```

Six panels, all seed-averaged, all showing the spread: retention heatmap, drop
heatmap, raw-return heatmap (with the random-agent floor as a reference row),
per-task forgetting curves with ±std bands, a summary-metrics panel, and per-task
drop bars with whiskers *and* the individual seed dots. Output goes to
`runs/figures/matrix_<group>.<fmt>`; `--bars` also writes the bar panel standalone
as `drop_bars_<group>.<fmt>`.

Filters: positional group names, or `--method` / `--sequence` / `--modality`.

Heatmap cells show the mean with ±std beneath, and cells averaged over fewer seeds
are marked `n=k`.

**Color differs from `../tools/visualize_matrix.py` on purpose.** Retention and
Drop are magnitudes, so they take a one-hue sequential ramp; the old `RdYlGn` is a
diverging map applied to a non-diverging quantity, puts a hue at its midpoint, and
is red-green — unreadable under the two commonest CVD types. Raw return genuinely
*is* signed, so it keeps a diverging map, centered on 0 with a neutral gray
midpoint.