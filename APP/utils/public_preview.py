from pathlib import Path

DEFAULT_KEEP_NAMES = frozenset({".gitkeep"})


def clear_public_preview_files(public_dir: Path, keep_names: frozenset[str] | None = None) -> None:
    keep = keep_names if keep_names is not None else DEFAULT_KEEP_NAMES
    if not public_dir.is_dir():
        return
    for path in public_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
