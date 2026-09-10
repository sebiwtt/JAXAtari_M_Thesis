# =============================================================================
# Generate the frozen random-order magnitude sequences (`randmag4`).
# =============================================================================
# The magnitude family asks whether the ORDER of a difficulty ladder matters, so
# it ships three orderings of the very same four rungs:
#
#     ascmag4   easy -> hard      (x2, x3, x4, x5)
#     descmag4  hard -> easy      (x5, x4, x3, x2)
#     randmag4  one fixed shuffle (this script)
#
# A random order is only a fair third condition if it is drawn ONCE and then
# frozen as part of the benchmark - redrawing per run would make it a different
# experiment each time, and picking it by hand would invite the suspicion that
# the ordering was chosen after seeing the results. So this script derives the
# permutation deterministically from a fixed master seed, writes it into the
# config as a normal sequence file, and can later re-derive it to prove the
# committed files are exactly what the seed produces:
#
#   python tools/gen_randmag4.py            # write config/sequence/<game>/<game>_randmag4.yaml
#   python tools/gen_randmag4.py --check    # verify the committed files, exit 1 on drift
#   python tools/gen_randmag4.py --print    # just show the permutations
#
# Reproducibility notes (the bits worth stating in a methodology section):
#
#   * The rungs are READ from <game>_ascmag4.yaml rather than hardcoded here, so
#     all three orderings are guaranteed to use the identical mod set. Change the
#     ascending ladder and the random order follows it.
#   * The shuffle uses a SHA-256 byte stream (`_HashRandom` below), not
#     random.shuffle or numpy. Python's and numpy's generators only promise
#     stream stability within a major version / for the legacy RandomState;
#     SHA-256 is fixed forever, so this script yields the same permutation on any
#     machine, any interpreter, any year, with no dependency pinning.
#   * Each game gets its OWN permutation, derived from its name. Reusing one
#     permutation across all six games would confound "order matters" with the
#     particular order drawn - one unlucky draw would bias every game the same
#     way. Six independent draws average that out.
#   * A draw equal to the ascending or the descending order is rejected and
#     redrawn (see `_draw_permutation`). Without this, randmag4 would have a
#     2/24 chance of silently duplicating a condition that is already in the
#     benchmark, leaving the study with two identical arms instead of three.
#     Rejection keeps the draw uniform over the 22 remaining permutations.
# =============================================================================

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

CRL_DIR = Path(__file__).resolve().parent.parent  # scripts/benchmarks/CRL
SEQUENCE_DIR = CRL_DIR / "config" / "sequence"

# The six games carrying a magnitude ladder.
GAMES = ["pong", "freeway", "breakout", "asteroids", "seaquest", "kangaroo"]

# --- Frozen parameters of the draw. Changing either re-rolls every sequence. ---
MASTER_SEED = 42
# Versioned so a future, deliberately different draw can coexist with this one in
# the thesis' history instead of silently overwriting it.
NAMESPACE = "jaxatari-crl-randmag4/v1"


class _HashRandom:
    """Deterministic byte stream from SHA-256, used as the shuffle's randomness.

    Uses the hash as a counter-mode stream: sha256(seed || counter) for
    counter = 0, 1, 2, ... Only stdlib hashlib, so the output is identical
    everywhere and for all time - which is the whole point of using it instead
    of `random`.
    """

    def __init__(self, seed_material: bytes):
        self._seed = seed_material
        self._counter = 0
        self._buf = b""

    def _next_byte(self) -> int:
        if not self._buf:
            self._buf = hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
        byte, self._buf = self._buf[0], self._buf[1:]
        return byte

    def below(self, n: int) -> int:
        """Uniform integer in [0, n), 1 <= n <= 256.

        Rejects the top partial bucket instead of taking `byte % n`, which would
        make the low indices slightly more likely (modulo bias).
        """
        if not 1 <= n <= 256:
            raise ValueError(f"below() supports 1..256, got {n}")
        limit = 256 - (256 % n)
        while True:
            byte = self._next_byte()
            if byte < limit:
                return byte % n

    def permutation(self, n: int) -> list[int]:
        """Fisher-Yates shuffle of range(n), walking i from n-1 down to 1."""
        out = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


def _draw_permutation(game: str, n: int, master_seed: int) -> tuple[list[int], int]:
    """The frozen permutation for one game, plus how many draws were rejected.

    Redraws while the permutation is the identity (== ascmag4) or the reversal
    (== descmag4); the stream keeps advancing, so each redraw is a fresh draw.
    """
    rng = _HashRandom(f"{NAMESPACE}|{game}|{master_seed}".encode())
    ascending = list(range(n))
    descending = ascending[::-1]
    rejected = 0
    while True:
        perm = rng.permutation(n)
        if perm != ascending and perm != descending:
            return perm, rejected
        rejected += 1


def _asc_ladder(game: str) -> list[str]:
    """The four mod keys of <game>_ascmag4, in ascending order."""
    path = SEQUENCE_DIR / game / f"{game}_ascmag4.yaml"
    if not path.exists():
        raise SystemExit(f"missing {path} - randmag4 is derived from the ascending ladder")
    cfg = yaml.safe_load(path.read_text())
    tasks = cfg["TASK_MODS"]
    if not tasks or tasks[0] != []:
        raise SystemExit(f"{path}: TASK_MODS[0] must be the base task (no mods)")
    ladder = []
    for i, mods in enumerate(tasks[1:], start=1):
        if len(mods) != 1:
            raise SystemExit(f"{path}: TASK_MODS[{i}]={mods} must hold exactly one mod")
        ladder.append(mods[0])
    return ladder


def render(game: str, master_seed: int = MASTER_SEED) -> tuple[str, list[int]]:
    """The full text of <game>_randmag4.yaml, and the permutation behind it."""
    ladder = _asc_ladder(game)
    perm, rejected = _draw_permutation(game, len(ladder), master_seed)
    ordered = [ladder[i] for i in perm]

    note = ""
    if rejected:
        plural = "s" if rejected > 1 else ""
        note = (f"#   redraws     : {rejected} (drew the ascending/descending order "
                f"{rejected} time{plural} first)\n")

    body = "\n".join(f'  - ["{mod}"]' for mod in ordered)
    return (
        "# @package _global_\n"
        "# AUTO-GENERATED by tools/gen_randmag4.py - do not edit by hand.\n"
        "#\n"
        f"# Random-order magnitude ladder: the same four rungs as {game}_ascmag4, shuffled\n"
        "# once under a fixed seed and then frozen as part of the benchmark.\n"
        f"#   master seed : {master_seed}\n"
        f'#   derivation  : SHA-256("{NAMESPACE}|{game}|{master_seed}") -> Fisher-Yates\n'
        f"#   permutation : {perm} (0-based indices into the ascending ladder)\n"
        "#   constraint  : redrawn if it equals the ascending or descending order\n"
        f"{note}"
        "# Verify with: python tools/gen_randmag4.py --check\n"
        f"ENV_ID: {game}\n"
        "SEQUENCE: randmag4\n"
        "TASK_MODS:\n"
        "  - []\n"
        f"{body}\n"
    ), perm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed files match the seed; exit 1 on drift")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the permutations without touching any file")
    ap.add_argument("--seed", type=int, default=MASTER_SEED,
                    help=f"master seed (default {MASTER_SEED}; changing it re-rolls every sequence)")
    ap.add_argument("--games", nargs="+", default=GAMES, help="restrict to these games")
    args = ap.parse_args()

    drift = []
    for game in args.games:
        text, perm = render(game, args.seed)
        path = SEQUENCE_DIR / game / f"{game}_randmag4.yaml"
        ladder = _asc_ladder(game)
        order = " -> ".join(ladder[i] for i in perm)

        if args.print_only:
            print(f"{game:10s} {str(perm):14s} {order}")
            continue

        if args.check:
            if not path.exists():
                drift.append(f"{path.relative_to(CRL_DIR)}: missing")
            elif path.read_text() != text:
                drift.append(f"{path.relative_to(CRL_DIR)}: does not match seed {args.seed}")
            else:
                print(f"ok   {game:10s} {perm}  {order}")
            continue

        path.write_text(text)
        print(f"wrote {path.relative_to(CRL_DIR)}  {perm}  {order}")

    if drift:
        print("\nDRIFT - the committed randmag4 files are not what the seed produces:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print("\nRe-run without --check to regenerate them.", file=sys.stderr)
        return 1
    if args.check:
        print(f"\nall {len(args.games)} randmag4 sequences reproduce from seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
