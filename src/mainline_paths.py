"""Path helpers shared by the compact mainline."""

from __future__ import annotations

from pathlib import Path


def redwood_partial_root(project_root: Path) -> Path:
    """Canonical location of the moved Redwood partial scans."""
    return Path(project_root) / "data" / "redwood" / "partial"


def redwood_ground_truth_root(project_root: Path) -> Path:
    """Canonical offline-only location of the moved Redwood ground truth."""
    return Path(project_root) / "data" / "redwood" / "gt"


def sample_dir(cfg, sample: str) -> Path:
    return Path(cfg.paths.output_dir).resolve() / str(sample)


def sample_file(cfg, sample: str, name: str) -> Path:
    return sample_dir(cfg, sample) / name


def model_path(cfg, key: str, default_relative: str) -> Path:
    configured = getattr(getattr(cfg, "models", {}), key, None)
    path = Path(configured or default_relative).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(cfg.paths.models_dir).resolve() / path).resolve()
