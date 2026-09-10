from pathlib import Path

from src.mainline_paths import (
    REGISTERED_PRIOR_FILENAME,
    POSTERIOR_PREDICTION_FILENAME,
    locate_mainline_prediction,
    locate_registered_prior,
    redwood_ground_truth_root,
    redwood_partial_root,
    registered_prior_path,
    posterior_prediction_path,
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


def test_current_mainline_prediction_uses_posterior_directory(tmp_path: Path) -> None:
    canonical = posterior_prediction_path(tmp_path, "01184")
    assert canonical == tmp_path / "01184" / POSTERIOR_PREDICTION_FILENAME
    assert locate_mainline_prediction(tmp_path, "01184") == canonical
