"""Strict, auditable contracts for scene-level instance completion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True)
class SceneInstance:
    """One GPT-segmented visible object in the original scene image."""

    instance_id: str
    label: str
    mask_path: Path
    layer: int = 0  # Larger values are foreground at mask-overlap resolution.

    def __post_init__(self) -> None:
        if not _INSTANCE_ID.fullmatch(self.instance_id):
            raise ValueError(
                "instance_id must be a safe non-empty file stem, got "
                f"{self.instance_id!r}"
            )
        if not self.label.strip():
            raise ValueError(f"instance {self.instance_id!r} needs a non-empty label")


@dataclass(frozen=True)
class SceneManifest:
    """Instance masks and labels linked to exactly one RGB scene image."""

    path: Path
    source_image: Path
    instances: tuple[SceneInstance, ...]
    schema_version: int = 1

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(item.instance_id for item in self.instances)


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"scene instance manifest is missing required key {key!r}")
    return mapping[key]


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_scene_manifest(path: Path, *, expected_source_image: Path | None = None) -> SceneManifest:
    """Load an agent-produced instance manifest without guessing any labels.

    The manifest is deliberately small and model-agnostic.  GPT image editing
    creates the binary masks; this loader only checks that a later geometry
    stage consumes exactly those saved artefacts.
    """
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene instance manifest must be a JSON object")
    version = int(payload.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"unsupported scene instance manifest schema {version}")
    base = path.parent
    source = _resolve(base, _required(payload, "source_image"))
    if expected_source_image is not None and source != Path(expected_source_image).resolve():
        raise ValueError(
            "manifest source_image does not match --scene-image: "
            f"{source} != {Path(expected_source_image).resolve()}"
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_instances = _required(payload, "instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("scene instance manifest needs a non-empty instances list")
    instances: list[SceneInstance] = []
    seen: set[str] = set()
    for raw in raw_instances:
        if not isinstance(raw, dict):
            raise ValueError("each scene instance must be a JSON object")
        item = SceneInstance(
            instance_id=str(_required(raw, "id")),
            label=str(_required(raw, "label")),
            mask_path=_resolve(base, _required(raw, "mask")),
            layer=int(raw.get("layer", 0)),
        )
        if item.instance_id in seen:
            raise ValueError(f"duplicate scene instance id {item.instance_id!r}")
        if not item.mask_path.is_file():
            raise FileNotFoundError(item.mask_path)
        seen.add(item.instance_id)
        instances.append(item)
    return SceneManifest(path=path, source_image=source, instances=tuple(instances), schema_version=version)


def write_gpt_instance_template(path: Path, *, source_image: Path, instances: list[dict[str, str]]) -> None:
    """Write an explicit fill-in template for the agent-mediated GPT mask step."""
    payload = {
        "schema_version": 1,
        "source_image": str(Path(source_image).resolve()),
        "instances": [
            {
                "id": str(item["id"]),
                "label": str(item["label"]),
                "mask": str(item.get("mask", f"masks/{item['id']}.png")),
                "layer": int(item.get("layer", 0)),
            }
            for item in instances
        ],
        "mask_contract": (
            "Each mask is a same-resolution binary PNG in source-image pixel coordinates; "
            "white foreground selects one physical object and black selects all other pixels."
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
