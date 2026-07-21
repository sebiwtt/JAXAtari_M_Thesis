# =============================================================================
# Launch ppo_crl_continual.py across multiple seed replicates, in parallel across GPUs.
# Mirrors run_all_pqn.py's subprocess-pool pattern; independent OS processes rather
# than pqn_agent.py's jax.vmap, since the continual loop does real disk I/O per task.
#
# Only SEED is passed per replicate - EVAL_SEED is left unset so it's derived from SEED
# (SEED * 12 + 1, see config/config.yaml) rather than reusing one identical eval seed
# across every replicate.
#
# Usage:
#   python tools/run_all_crl_seeds.py --gpus 0,1,2,3 --seeds 0,1,2,3,4 --sequence pong_dyn4 --method ewc --modality oc
#   python tools/run_all_crl_seeds.py --gpus 0 --seeds 0,1,2 -- TOTAL_TIMESTEPS=1000000
# (anything after the flags above is forwarded verbatim to ppo_crl_continual.py)
# =============================================================================

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

DEFAULT_SEEDS = [0, 1, 2]

# Concurrent processes per GPU; keep at 1 unless a seed run comfortably fits multiple
# times in GPU memory (default NUM_ENVS=8192 does not).
WORKERS_PER_GPU = 1


def worker(gpu_id: str, worker_id: int, task_queue: "queue.Queue", composition: list, extra_args: list):
    """Continuously pull seeds off the queue and run one full continual sweep per seed."""
    while not task_queue.empty():
        try:
            seed = task_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[GPU {gpu_id} | Worker {worker_id}] Starting seed {seed} ({' '.join(composition)})...")

        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            "uv", "run", "scripts/benchmarks/CRL/ppo_crl_continual.py",
            *composition,
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
    parser.add_argument("--sequence", type=str, default="pong_dyn4", help="Hydra sequence group (config/sequence/<name>.yaml).")
    parser.add_argument("--method", type=str, default="ft", help="Hydra method group (config/method/<name>.yaml): ft, ewc, agem, packnet.")
    parser.add_argument("--modality", type=str, default="oc", help="Hydra modality group (config/modality/<name>.yaml): oc, pixel.")

    # Anything unrecognized is forwarded verbatim to ppo_crl_continual.py.
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
                composition = [f"sequence={args.sequence}", f"method={args.method}", f"modality={args.modality}"]
                executor.submit(worker, gpu_id, worker_id, task_queue, composition, extra_args)

    task_queue.join()
    print("All seed runs finished. Aggregate them with aggregate_crl_seeds.py, e.g.:")
    print("  python scripts/benchmarks/CRL/aggregate_crl_seeds.py --glob 'runs/<ENV_ID>_<EXP_NAME>_<oc|pixel>_*'")
    print("  (the exact prefix is printed by each ppo_crl_continual.py run as '[CRL] matrix saved to ...')")


if __name__ == "__main__":
    main()
