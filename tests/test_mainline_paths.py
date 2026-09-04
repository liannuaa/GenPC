from pathlib import Path

from src.mainline_paths import (
    GAUSSIAN_PREDICTION_FILENAME,
    REGISTERED_PRIOR_FILENAME,
    gaussian_prediction_path,
    locate_gaussian_prediction,
    locate_registered_prior,
    redwood_ground_truth_root,
    redwood_partial_root,
    registered_prior_path,
)


def test_redwood_roots_follow_the_public_data_contract() -> None:
    project_root = Path("project")
    assert redwood_partial_root(project_root) == project_root / "data" / "redwood" / "partial"
    assert redwood_ground_truth_root(project_root) == project_root / "data" / "redwood" / "gt"


def test_registered_prior_keeps_legacy_resume_preference(tmp_path: Path) -> None:
    canonical = registered_prior_path(tmp_path, "01184")
    assert canonical == tmp_path / "01184" / "final" / REGISTERED_PRIOR_FILENAME
    assert locate_registered_prior(tmp_path, "01184") == canonical

    legacy = tmp_path / "01184" / REGISTERED_PRIOR_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert locate_registered_prior(tmp_path, "01184") == legacy


def test_gaussian_prediction_keeps_legacy_flat_fallback(tmp_path: Path) -> None:
    canonical = gaussian_prediction_path(tmp_path, "01184")
    assert canonical == tmp_path / "01184" / "decoded" / GAUSSIAN_PREDICTION_FILENAME
    assert locate_gaussian_prediction(tmp_path, "01184") == tmp_path / "01184" / GAUSSIAN_PREDICTION_FILENAME

    canonical.parent.mkdir(parents=True)
    canonical.touch()
    assert locate_gaussian_prediction(tmp_path, "01184") == canonical
