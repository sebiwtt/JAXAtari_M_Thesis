# =============================================================================
# Sequence config-group routing
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
