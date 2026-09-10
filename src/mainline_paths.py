"""Shared, stable path and artifact contracts for the fixed mainline.

The public runners deliberately import their common sample set and final-file
names from here. This keeps a full rerun and an offline audit pointed at the
same artifacts without coupling numerical code to an experiment directory.
"""

from __future__ import annotations

from pathlib import Path


REDWOOD10_SAMPLE_IDS = (
    "01184", "05117", "05452", "06127", "06145",
    "06188", "06830", "07136", "07306", "09639",
)

REGISTERED_PRIOR_FILENAME = "camera1_amplified_registered_100k.ply"
GAUSSIAN_PREDICTION_FILENAME = "partial_anchored_gaussian_decoded_100k.ply"
TRELLIS_PREDICTION_FILENAME = "complete_100k.ply"


def redwood_partial_root(project_root: Path) -> Path:
    """Canonical location of the moved Redwood partial scans."""
    return Path(project_root) / "data" / "redwood" / "partial"


def redwood_ground_truth_root(project_root: Path) -> Path:
    """Canonical offline-only location of the moved Redwood ground truth."""
    return Path(project_root) / "data" / "redwood" / "gt"


def registered_prior_path(registration_root: Path, sample: str) -> Path:
    """Canonical final Camera-1 registration artifact for ``sample``."""
    return Path(registration_root) / str(sample) / "final" / REGISTERED_PRIOR_FILENAME


def locate_registered_prior(registration_root: Path, sample: str) -> Path:
    """Return a registered prior, retaining the historical flat-path fallback.

    Early compact runs wrote the final PLY directly under ``<sample>``. The
    runner has always preferred that path when present, so retaining this order
    preserves resume behaviour for existing results.
    """
    root = Path(registration_root) / str(sample)
    legacy = root / REGISTERED_PRIOR_FILENAME
    return legacy if legacy.is_file() else registered_prior_path(registration_root, sample)


def gaussian_prediction_path(prediction_root: Path, sample: str) -> Path:
    """Canonical decoded 100k Gaussian prediction artifact for ``sample``."""
    return Path(prediction_root) / str(sample) / "decoded" / GAUSSIAN_PREDICTION_FILENAME


def locate_gaussian_prediction(prediction_root: Path, sample: str) -> Path:
    """Return a decoded prediction, retaining the historical flat-path fallback."""
    canonical = gaussian_prediction_path(prediction_root, sample)
    if canonical.is_file():
        return canonical
    return Path(prediction_root) / str(sample) / GAUSSIAN_PREDICTION_FILENAME


def trellis_prediction_path(run_root: Path, sample: str) -> Path:
    """Canonical fusion-free prediction produced by the object mainline."""
    return Path(run_root) / "final" / str(sample) / TRELLIS_PREDICTION_FILENAME


def locate_mainline_prediction(prediction_root: Path, sample: str) -> Path:
    """Locate the current prediction while accepting a direct ``final`` root."""
    root = Path(prediction_root)
    canonical = trellis_prediction_path(root, sample)
    if canonical.is_file():
        return canonical
    direct = root / str(sample) / TRELLIS_PREDICTION_FILENAME
    return direct if direct.is_file() else canonical


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
