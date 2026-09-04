"""Path helpers used by the semantic stage."""

from __future__ import annotations

from pathlib import Path


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
