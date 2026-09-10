"""Artifact contract for the fusion-free GenPC++ object mainline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

from src.trellis_multiview_probe import (
    compose_equal_view_board,
    normalise_edited_view_framing,
    split_equal_view_board,
)


VIEWS = ("front", "side", "back")
FINAL_PREDICTION_FILENAME = "complete_100k.ply"


class ExternalArtifactRequired(RuntimeError):
    """Raised when a resumable external image-edit checkpoint is missing."""

    def __init__(self, message: str, paths: Iterable[Path]):
        self.paths = tuple(Path(path) for path in paths)
        super().__init__(message)


@dataclass(frozen=True)
class ObjectMainlineLayout:
    """Canonical paths for one object, independent of its category or dataset."""

    root: Path
    sample: str

    @property
    def partial_root(self) -> Path:
        return self.root / "inputs" / "partial"

    @property
    def partial(self) -> Path:
        return self.partial_root / f"{self.sample}.ply"

    @property
    def camera_root(self) -> Path:
        return self.root / "inputs" / "camera"

    @property
    def camera_dir(self) -> Path:
        return self.camera_root / self.sample

    @property
    def camera(self) -> Path:
        return self.camera_dir / "camera.pth"

    @property
    def semantic(self) -> Path:
        return self.camera_dir / "img.png"

    @property
    def pixal_root(self) -> Path:
        return self.root / "inputs" / "pixal"

    @property
    def pixal_dir(self) -> Path:
        return self.pixal_root / self.sample

    @property
    def clarity_image(self) -> Path:
        return self.pixal_dir / "gpt_image.png"

    @property
    def clarity_prompt(self) -> Path:
        return self.pixal_dir / "prompt.txt"

    @property
    def pixal_mesh(self) -> Path:
        return self.pixal_dir / "pixal3d.glb"

    @property
    def pixal_carrier(self) -> Path:
        return self.pixal_dir / "pixal3d_sampled_100k.ply"

    @property
    def pixal_registration_root(self) -> Path:
        return self.root / "pixal_registration"

    @property
    def registered_pixal(self) -> Path:
        return (
            self.pixal_registration_root / self.sample / "final"
            / "camera1_amplified_registered_100k.ply"
        )

    @property
    def multiview(self) -> Path:
        return self.root / "multiview" / self.sample

    @property
    def render_dir(self) -> Path:
        return self.multiview / "render"

    @property
    def render_manifest(self) -> Path:
        return self.render_dir / "render_manifest.json"

    @property
    def render_board(self) -> Path:
        return self.render_dir / "pixal_registered_front_side_back_board.png"

    def render_view(self, view: str) -> Path:
        return self.render_dir / f"pixal_registered_{view}.png"

    @property
    def evidence_dir(self) -> Path:
        return self.multiview / "evidence"

    @property
    def evidence_board(self) -> Path:
        return self.evidence_dir / "shared_local_residual_board.png"

    @property
    def stage1_prompt(self) -> Path:
        return self.evidence_dir / "stage1_shared_low_frequency_prompt.txt"

    @property
    def edits_dir(self) -> Path:
        return self.multiview / "edits"

    @property
    def stage1_board(self) -> Path:
        return self.edits_dir / "shared_low_frequency_front_side_back.png"

    @property
    def stage1_views_dir(self) -> Path:
        return self.edits_dir / "stage1_views"

    def stage1_view(self, view: str) -> Path:
        return self.stage1_views_dir / f"trellis_condition_{view}.png"

    @property
    def local_raw_dir(self) -> Path:
        return self.edits_dir / "local_raw"

    def local_raw_view(self, view: str) -> Path:
        return self.local_raw_dir / f"{view}.png"

    @property
    def local_views_dir(self) -> Path:
        return self.edits_dir / "local"

    def local_view(self, view: str) -> Path:
        return self.local_views_dir / f"{view}.png"

    @property
    def conditions_dir(self) -> Path:
        return self.multiview / "conditions"

    def condition(self, view: str) -> Path:
        return self.conditions_dir / f"{view}.png"

    @property
    def condition_board(self) -> Path:
        return self.conditions_dir / "front_side_back.png"

    @property
    def trellis_dir(self) -> Path:
        return self.root / "trellis" / self.sample

    @property
    def trellis_mesh(self) -> Path:
        return self.trellis_dir / "trellis_mesh_raw.glb"

    @property
    def trellis_carrier(self) -> Path:
        return self.trellis_dir / "trellis_mesh_sampled_100k.ply"

    @property
    def registration_root(self) -> Path:
        return self.root / "trellis_registration" / self.sample

    def selection_dir(self, view: str) -> Path:
        return self.registration_root / "view_selection" / view

    def selection_json(self, view: str) -> Path:
        return self.selection_dir(view) / "trellis_camera1_view.json"

    @property
    def semantic_capture_dir(self) -> Path:
        return self.registration_root / "semantic_capture"

    @property
    def camera_capture_dir(self) -> Path:
        return self.registration_root / "camera_capture"

    @property
    def camera_polish_dir(self) -> Path:
        return self.registration_root / "camera_polish"

    @property
    def final_registration_dir(self) -> Path:
        return self.registration_root / "final"

    @property
    def prediction(self) -> Path:
        return self.root / "final" / self.sample / FINAL_PREDICTION_FILENAME

    @property
    def run_manifest(self) -> Path:
        return self.root / "manifests" / f"{self.sample}.json"

    def status(self) -> dict[str, bool]:
        """Report resumable checkpoints without reading any GT artifact."""
        groups = {
            "partial": (self.partial,),
            "semantic": (self.camera, self.semantic),
            "clarity": (self.clarity_image, self.clarity_prompt),
            "pixal_prior": (self.pixal_mesh, self.pixal_carrier),
            "pixal_registration": (self.registered_pixal,),
            "multiview_evidence": (
                self.render_manifest, self.render_board, self.evidence_board,
                self.stage1_prompt,
            ),
            "shared_edit": (self.stage1_board,),
            "local_edits": tuple(self.local_raw_view(view) for view in VIEWS),
            "camera_conditions": tuple(self.condition(view) for view in VIEWS),
            "trellis_prior": (self.trellis_mesh, self.trellis_carrier),
            "trellis_registration": (
                self.final_registration_dir / "trellis_registered_100k.ply",
            ),
            "prediction": (self.prediction,),
        }
        return {name: all(path.is_file() for path in paths) for name, paths in groups.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_input_once(source: Path, destination: Path) -> Path:
    """Copy an input without silently replacing a different saved artifact."""
    source, destination = Path(source).resolve(), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.is_file():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"saved input differs from requested source: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def materialize_camera_conditions(layout: ObjectMainlineLayout) -> dict:
    """Convert two external edit checkpoints into calibrated TRELLIS views."""
    if not layout.stage1_board.is_file():
        raise ExternalArtifactRequired(
            "shared low-frequency three-view edit is required",
            (layout.stage1_board,),
        )
    stage1 = split_equal_view_board(layout.stage1_board, layout.stage1_views_dir, VIEWS)
    missing = [layout.local_raw_view(view) for view in VIEWS if not layout.local_raw_view(view).is_file()]
    if missing:
        raise ExternalArtifactRequired(
            "residual-only per-view edits are required", missing,
        )

    records = {}
    conditions = []
    for view, stage1_path in zip(VIEWS, stage1):
        local_path = layout.local_view(view)
        local_record = normalise_edited_view_framing(
            layout.local_raw_view(view), stage1_path, local_path,
        )
        condition = layout.condition(view)
        contract_record = normalise_edited_view_framing(
            local_path, layout.render_view(view), condition,
        )
        conditions.append(condition)
        records[view] = {"local": local_record, "camera_contract": contract_record}
    compose_equal_view_board(conditions, layout.condition_board)
    record = {
        "method": "shared_low_frequency_then_local_residual_edit",
        "ground_truth_used": False,
        "views": records,
        "condition_board": str(layout.condition_board.resolve()),
    }
    path = layout.conditions_dir / "condition_manifest.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def publish_prediction(layout: ObjectMainlineLayout) -> Path:
    source = layout.final_registration_dir / "trellis_registered_100k.ply"
    if not source.is_file():
        raise FileNotFoundError(source)
    if layout.prediction.is_file():
        if sha256(source) != sha256(layout.prediction):
            raise FileExistsError(
                f"published prediction differs from current registration: {layout.prediction}"
            )
        return layout.prediction
    layout.prediction.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, layout.prediction)
    return layout.prediction


def write_manifest(layout: ObjectMainlineLayout, *, object_type: str, commands: list[list[str]]) -> Path:
    """Record the no-GT execution contract and hashes of available key files."""
    key_paths = {
        "partial": layout.partial,
        "semantic": layout.semantic,
        "clarity_image": layout.clarity_image,
        "registered_pixal": layout.registered_pixal,
        "edited_front": layout.condition("front"),
        "edited_side": layout.condition("side"),
        "edited_back": layout.condition("back"),
        "trellis_carrier": layout.trellis_carrier,
        "prediction": layout.prediction,
    }
    previous_commands: list[list[str]] = []
    if layout.run_manifest.is_file():
        previous = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
        if previous.get("sample") == layout.sample:
            previous_commands = list(previous.get("commands", []))
    record = {
        "method": "genpc_plus_trellis_regeneration_mainline",
        "sample": layout.sample,
        "object_type": object_type,
        "ground_truth_used": False,
        "fusion_used": False,
        "status": layout.status(),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in key_paths.items() if path.is_file()
        },
        "commands": previous_commands + commands,
    }
    layout.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.run_manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return layout.run_manifest
