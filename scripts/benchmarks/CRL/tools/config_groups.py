# =============================================================================
# Sequence config-group routing
# =============================================================================
# config/sequence/ is organized one folder per game, with the file names left
# fully qualified:
#
#     config/sequence/pong/pong_dyn4.yaml
#
# so hydra's own name for that option is "pong/pong_dyn4". Everything here -
# scripts, campaign manifests, the README - spells sequences the flat way
# ("sequence=pong_dyn4"), so `resolve_sequence` maps the flat name onto the
# nested one by looking up the file, and `rewrite_sequence_argv` applies that to
# sys.argv just before a hydra entry point parses it. Nothing else has to change:
# both spellings work everywhere, and a flat config/sequence/<name>.yaml still
# takes precedence if one exists.
# =============================================================================

from pathlib import Path

SEQUENCE_DIR = Path(__file__).resolve().parent.parent / "config" / "sequence"


def resolve_sequence(name: str) -> str:
    """Flat sequence name -> the group option hydra expects.

    "pong_dyn4" -> "pong/pong_dyn4". Already-nested names, flat files, and
    unknown names are returned unchanged (hydra then raises its own error, which
    lists the valid options).
    """
    if "/" in name or (SEQUENCE_DIR / f"{name}.yaml").exists():
        return name
    matches = sorted(SEQUENCE_DIR.glob(f"*/{name}.yaml"))
    if len(matches) > 1:
        folders = ", ".join(m.parent.name for m in matches)
        raise SystemExit(f"sequence '{name}' is ambiguous - it exists in: {folders}")
    if matches:
        return f"{matches[0].parent.name}/{name}"
    return name


def sequence_yaml_path(name: str) -> Path:
    """Path of a sequence config, accepting either spelling."""
    return SEQUENCE_DIR / f"{resolve_sequence(name)}.yaml"


def rewrite_sequence_argv(argv: list[str]) -> None:
    """In-place rewrite of any `sequence=<flat>` override in argv. Call right
    before handing control to a hydra entry point."""
    for i, arg in enumerate(argv):
        for prefix in ("sequence=", "+sequence=", "++sequence="):
            if arg.startswith(prefix):
                argv[i] = prefix + resolve_sequence(arg[len(prefix):])
                break
