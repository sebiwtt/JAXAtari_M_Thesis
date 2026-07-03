# =============================================================================
# Launch ppo_crl_continual.py across multiple seed replicates, in parallel across GPUs.
# =============================================================================
# Mirrors scripts/benchmarks/run_all_pqn.py's pattern (subprocess pool + worker queue +
# CUDA_VISIBLE_DEVICES pinning), adapted for this CRL harness: one seed sweep of one
# alg config, not an env x seed x config grid. The continual training loop is a Python
# host-side loop over tasks doing real disk I/O between them (checkpoint writes,
# evaluate() reading checkpoints back), so it can't be vmapped the way pqn_agent.py
# vmaps its NUM_SEEDS - independent OS processes is the only way to parallelize seeds
# here.
#
# Only SEED is varied between replicates; EVAL_SEED is left untouched (whatever the alg
# config sets) so every replicate is scored under an identical eval protocol - otherwise
# eval-rollout noise would be a confound mixed into your seed-to-seed variance. Each
# replicate's outputs already land in a SEED-namespaced runs/{...}_{SEED}/ dir (see
# ppo_crl_continual.py), so parallel replicates can't collide on disk.
#
# After all seeds finish, aggregate their runs/.../matrix.json files with
# aggregate_crl_seeds.py.
#
# Usage:
#   python run_all_crl_seeds.py --gpus 0,1,2,3 --seeds 0,1,2,3,4
#   python run_all_crl_seeds.py --gpus 0 --seeds 0,1,2 -- alg.TOTAL_TIMESTEPS=1000000
# (anything after the flags above is forwarded verbatim to ppo_crl_continual.py, e.g.
# Hydra overrides for TASK_MODS/EVAL_EPISODES/etc.)
# =============================================================================

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Concurrent processes per GPU. Keep at 1 unless a single seed run comfortably fits
# multiple times in one GPU's memory (NUM_ENVS=8192 by default does not).
WORKERS_PER_GPU = 1


def worker(gpu_id: str, worker_id: int, task_queue: "queue.Queue", alg_config: str, extra_args: list):
    """Continuously pull seeds off the queue and run one full continual sweep per seed."""
    while not task_queue.empty():
        try:
            seed = task_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[GPU {gpu_id} | Worker {worker_id}] Starting seed {seed} (alg={alg_config})...")

        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            "uv", "run", "scripts/benchmarks/CRL/ppo_crl_continual.py",
            f"+alg={alg_config}",
            f"SEED={seed}",
        ] + extra_args

        try:
            subprocess.run(cmd, env=env_vars, check=True)
            print(f"[GPU {gpu_id} | Worker {worker_id}] Finished seed {seed}.")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id} | Worker {worker_id}] Failed seed {seed} with exit code {e.returncode}.")
        finally:
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Run ppo_crl_continual.py across multiple seeds concurrently, one process per GPU worker."
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use (e.g. '0,1,2,3').")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed list, e.g. '0,1,2,3,4'.",
    )
    parser.add_argument("--alg", type=str, default="ppo_crl_continual", help="Hydra alg config name (config/alg/<name>.yaml).")

    # Parse known args; anything else gets forwarded verbatim to ppo_crl_continual.py
    # (e.g. `alg.TOTAL_TIMESTEPS=...`, `alg.TASK_MODS=...`).
    args, extra_args = parser.parse_known_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not gpus:
        print("Error: No GPUs specified.")
        return
    if not seeds:
        print("Error: No seeds specified.")
        return

    task_queue = queue.Queue()
    for seed in seeds:
        task_queue.put(seed)

    print(f"Starting {len(seeds)} seed run(s) {seeds} across {len(gpus)} GPU(s): {gpus} ({WORKERS_PER_GPU} worker(s) per GPU)")
    print(f"Extra args for ppo_crl_continual.py: {' '.join(extra_args) if extra_args else 'None'}")

    total_workers = len(gpus) * WORKERS_PER_GPU
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        for gpu_id in gpus:
            for worker_id in range(WORKERS_PER_GPU):
                executor.submit(worker, gpu_id, worker_id, task_queue, args.alg, extra_args)

    task_queue.join()
    print("All seed runs finished. Aggregate them with aggregate_crl_seeds.py, e.g.:")
    print("  python scripts/benchmarks/CRL/aggregate_crl_seeds.py --glob 'runs/<ENV_ID>_<EXP_NAME>_<oc|pixel>_*'")
    print("  (the exact prefix is printed by each ppo_crl_continual.py run as '[CRL] matrix saved to ...')")


if __name__ == "__main__":
    main()
