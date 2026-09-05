"""Convert GPT binary masks into object partials in a shared camera frame."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.scene_completion.contracts import SceneManifest
from src.scene_completion.io import write_colored_points
from src.scene_completion.scene_moge import SceneMoGeObservation


def _read_binary_mask(path: Path, *, image_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        # GPT image output is ordinarily RGB even for a deliberately black/white
        # mask.  Accept only near-monochrome RGB rather than accidentally
        # thresholding a coloured preview or an opaque checkerboard cutout.
        if mask.shape[-1] == 4:
            mask = mask[..., 3]
        elif mask.shape[-1] == 3:
            spread = np.abs(mask.astype(np.int16).max(axis=-1) - mask.astype(np.int16).min(axis=-1))
            if int(spread.max()) > 16:
                raise ValueError(
                    f"RGB mask {path} is not monochrome; request a strict black/white GPT mask "
                    "instead of a coloured cutout."
                )
            mask = np.rint(mask.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        else:
            raise ValueError(f"mask must be greyscale, monochrome RGB or RGBA, got {path} with shape {mask.shape}")
    if tuple(mask.shape[:2]) != tuple(image_hw):
        raise ValueError(
            f"mask {path} is {tuple(mask.shape[:2])}, but source image is {tuple(image_hw)}. "
            "GPT masks must stay in original scene-image coordinates."
        )
    return (mask > 127).astype(np.uint8)


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(np.uint8, copy=True)
    size = int(pixels) * 2 + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8), iterations=1)


def _bbox(mask: np.ndarray, *, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("cannot crop an empty object mask")
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + 1 + padding),
        min(height, int(ys.max()) + 1 + padding),
    )


def _masked_crop(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bbox
    crop = image[top:bottom, left:right].copy()
    crop_mask = mask[top:bottom, left:right].astype(bool)
    # Pure white is compatible with Pixal preprocessing and makes the GPT
    # completion request unambiguous.  The original RGB foreground is exact.
    crop[~crop_mask] = 255
    return crop


def _scene_pixels_in_mask(pixel_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.asarray(pixel_xy, dtype=np.float64)).astype(np.int64)
    height, width = mask.shape
    inside = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    keep = np.zeros(len(rounded), dtype=bool)
    ids = np.where(inside)[0]
    keep[ids] = mask[rounded[ids, 1], rounded[ids, 0]] > 0
    return keep


def extract_scene_instances(
    manifest: SceneManifest,
    observation: SceneMoGeObservation,
    *,
    output_root: Path,
    erosion_pixels: int = 1,
    crop_padding: int = 24,
    min_mask_pixels: int = 512,
    min_moge_points: int = 128,
) -> dict:
    """Write scene-frame partials and visual artifacts for all GPT instances.

    No object is centered, normalized, or transformed in this function.  A
    partial is simply a mask-indexed subset of the common scene MoGe cloud, so
    the unmodified single-object registration naturally returns a completion
    in the common scene-camera coordinate system.
    """
    output_root = Path(output_root).resolve()
    image = np.asarray(Image.open(manifest.source_image).convert("RGB"), dtype=np.uint8)
    if tuple(image.shape[:2]) != observation.image_hw:
        raise ValueError(
            f"source image shape {tuple(image.shape[:2])} does not match scene MoGe {observation.image_hw}"
        )
    partial_root = output_root / "inputs" / "partial"
    instance_root = output_root / "instances"
    mask_root = output_root / "masks"
    partial_root.mkdir(parents=True, exist_ok=True)
    instance_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    raw_masks: dict[str, np.ndarray] = {}
    for instance in manifest.instances:
        mask = _read_binary_mask(instance.mask_path, image_hw=observation.image_hw)
        if int(mask.sum()) < int(min_mask_pixels):
            raise ValueError(
                f"{instance.instance_id}: mask has only {int(mask.sum())} pixels; "
                f"minimum is {min_mask_pixels}"
            )
        raw_masks[instance.instance_id] = mask

    # GPT masks may overlap at physical occlusions or when a background object
    # leaks through a foreground silhouette.  Resolve every overlap once in
    # source-image space before selecting MoGe points: a larger `layer` means
    # physically closer to the camera; ties retain the manifest order.  This
    # avoids contaminating a complete-object prior with points from its
    # foreground occluder while retaining a fully shared scene camera frame.
    resolved_masks: dict[str, np.ndarray] = {}
    claimed = np.zeros(observation.image_hw, dtype=bool)
    ordering = sorted(enumerate(manifest.instances), key=lambda pair: (-pair[1].layer, pair[0]))
    for _, instance in ordering:
        resolved = raw_masks[instance.instance_id].astype(bool) & ~claimed
        if int(resolved.sum()) < int(min_mask_pixels):
            raise ValueError(
                f"{instance.instance_id}: layer-aware occlusion resolution retained only "
                f"{int(resolved.sum())} pixels; improve its GPT mask or layer assignment"
            )
        resolved_masks[instance.instance_id] = resolved.astype(np.uint8)
        claimed |= resolved

    union = np.zeros(observation.image_hw, dtype=np.uint8)
    records: list[dict] = []
    for instance in manifest.instances:
        raw_mask = raw_masks[instance.instance_id]
        original_mask = resolved_masks[instance.instance_id]
        mask = _erode(original_mask, int(erosion_pixels))
        # A very thin object can vanish under boundary protection.  Preserve
        # its exact GPT mask instead of dropping a physically valid instance.
        if int(mask.sum()) < int(min_mask_pixels):
            mask = original_mask
        selected = _scene_pixels_in_mask(observation.pixel_xy, mask)
        # The un-eroded, layer-resolved mask is a separate scene-layout anchor:
        # its original MoGe bbox records where the physical object sits in the
        # source scene.  Registration may use the eroded subset to avoid noisy
        # boundaries, but erosion must never move or shrink the final object.
        layout_anchor = _scene_pixels_in_mask(observation.pixel_xy, original_mask)
        if int(selected.sum()) < int(min_moge_points):
            raise ValueError(
                f"{instance.instance_id}: mask retained only {int(selected.sum())} scene MoGe points; "
                f"minimum is {min_moge_points}"
            )
        bbox = _bbox(original_mask, padding=int(crop_padding))
        target = instance_root / instance.instance_id
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_root / f"{instance.instance_id}.png"), original_mask * 255)
        cv2.imwrite(str(target / "eroded_mask.png"), mask * 255)
        Image.fromarray(_masked_crop(image, original_mask, bbox)).save(target / "masked_crop.png")
        selected_points = observation.points[selected]
        selected_colors = observation.colors[selected]
        anchor_points = observation.points[layout_anchor]
        anchor_colors = observation.colors[layout_anchor]
        write_colored_points(
            partial_root / f"{instance.instance_id}.ply",
            selected_points, selected_colors,
        )
        write_colored_points(
            target / "visible_partial_scene_frame.ply",
            selected_points, selected_colors,
        )
        write_colored_points(
            target / "layout_anchor_scene_frame.ply",
            anchor_points, anchor_colors,
        )
        record = {
            "id": instance.instance_id,
            "label": instance.label,
            "layer": int(instance.layer),
            "source_mask": str((mask_root / f"{instance.instance_id}.png").resolve()),
            "eroded_mask": str((target / "eroded_mask.png").resolve()),
            "masked_crop": str((target / "masked_crop.png").resolve()),
            "partial": str((partial_root / f"{instance.instance_id}.ply").resolve()),
            "layout_anchor": str((target / "layout_anchor_scene_frame.ply").resolve()),
            "bbox_xyxy": list(bbox),
            "raw_mask_pixels": int(raw_mask.sum()),
            "mask_pixels": int(original_mask.sum()),
            "occlusion_excluded_pixels": int(raw_mask.sum() - original_mask.sum()),
            "eroded_mask_pixels": int(mask.sum()),
            "scene_moge_points": int(selected.sum()),
            "layout_anchor_scene_moge_points": int(layout_anchor.sum()),
            "coordinate_frame": "shared_pixal_moge_scene_camera",
        }
        (target / "instance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records.append(record)
        union = np.maximum(union, original_mask)

    cv2.imwrite(str(output_root / "masks" / "all_instances_union.png"), union * 255)
    background = ~_scene_pixels_in_mask(observation.pixel_xy, union)
    # A tightly tiled collection of instance masks can legitimately claim
    # every visible MoGe point.  There is no empty-point-cloud representation
    # in the PLY writer, so omit this optional diagnostic instead of treating
    # a fully explained scene as an extraction failure.
    context_path = output_root / "scene_context_unmasked_visible.ply"
    if int(background.sum()):
        write_colored_points(context_path, observation.points[background], observation.colors[background])
    summary = {
        "method": "gpt_mask_indexed_pixal_moge_scene_instance_extraction",
        "source_image": str(manifest.source_image),
        "scene_moge_coordinate_frame": "shared_pixal_moge_scene_camera",
        "mask_erosion_pixels": int(erosion_pixels),
        "crop_padding_pixels": int(crop_padding),
        "instances": records,
        "unmasked_scene_context_points": int(background.sum()),
        "unmasked_scene_context": str(context_path.resolve()) if int(background.sum()) else None,
        "mask_union_pixels": int(union.sum()),
        "overlap_contract": (
            "source-image overlap is deterministically assigned to the highest layer; ties use manifest order. "
            "This uses only saved GPT masks and never GT or sample-specific routing"
        ),
    }
    (output_root / "scene_instance_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
