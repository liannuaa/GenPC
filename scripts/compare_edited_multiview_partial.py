#!/usr/bin/env python3
"""Compare a camera-locked edited view board with projected partial evidence.

The comparison is deliberately one-sided: observed partial pixels are positive
support, while missing pixels remain unknown.  No ground truth is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.multiview_partial_evidence import VIEW_ORDER, compose_grid


def _split_board(path: Path, resolution: int) -> dict[str, np.ndarray]:
    image = Image.open(path).convert("RGB")
    if image.width % 2 or image.height % 2:
        raise ValueError(f"board dimensions must be divisible by two: {path}")
    width, height = image.width // 2, image.height // 2
    panels: dict[str, np.ndarray] = {}
    for index, name in enumerate(VIEW_ORDER):
        left, top = (index % 2) * width, (index // 2) * height
        panel = image.crop((left, top, left + width, top + height)).resize(
            (resolution, resolution), Image.Resampling.LANCZOS,
        )
        panels[name] = np.asarray(panel, dtype=np.uint8)
    return panels


def _object_mask(rgb: np.ndarray) -> np.ndarray:
    # Images are generated on a white background.  A small closing operation
    # removes antialias pinholes without imposing category/part assumptions.
    colour_distance = np.max(255 - rgb.astype(np.int16), axis=2)
    mask = (colour_distance > 24).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def _support_mask(evidence: np.ndarray) -> np.ndarray:
    background = np.full_like(evidence, 18)
    return np.max(np.abs(evidence.astype(np.int16) - background), axis=2) > 16


def _support_metrics(object_mask: np.ndarray, support: np.ndarray) -> dict[str, float | int]:
    count = int(np.count_nonzero(support))
    if count == 0:
        return {
            "support_pixels": 0,
            "coverage_0px": 0.0,
            "coverage_3px": 0.0,
            "outside_distance_mean_px": 0.0,
            "outside_distance_p90_px": 0.0,
        }
    dilated = cv2.dilate(object_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    distance = cv2.distanceTransform((~object_mask).astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[support]
    return {
        "support_pixels": count,
        "coverage_0px": float(np.mean(object_mask[support])),
        "coverage_3px": float(np.mean(dilated[support])),
        "outside_distance_mean_px": float(np.mean(values)),
        "outside_distance_p90_px": float(np.quantile(values, 0.90)),
    }


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (8, 8), (142, 37), (20, 20, 20), -1)
    cv2.putText(output, text.upper(), (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited-board", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-board", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    manifest = json.loads(args.evidence_manifest.read_text(encoding="utf-8"))
    evidence_records = {item["name"]: item for item in manifest["views"]}
    edited = _split_board(args.edited_board, args.resolution)
    reference = (
        _split_board(args.reference_board, args.resolution)
        if args.reference_board is not None else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: list[Path] = []
    residual_paths: list[Path] = []
    vector_paths: list[Path] = []
    records: dict[str, dict] = {}
    for name in VIEW_ORDER:
        evidence = np.asarray(
            Image.open(evidence_records[name]["partial_evidence"]).convert("RGB").resize(
                (args.resolution, args.resolution), Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        support = _support_mask(evidence)
        edited_mask = _object_mask(edited[name])

        overlay = edited[name].copy()
        overlay[support] = np.rint(
            0.25 * overlay[support].astype(np.float32)
            + 0.75 * evidence[support].astype(np.float32)
        ).astype(np.uint8)
        overlay = _label(overlay, name)
        overlay_path = args.output_dir / f"edited_partial_overlay_{name}.png"
        Image.fromarray(overlay).save(overlay_path)
        overlay_paths.append(overlay_path)

        # Cyan means observed partial support currently outside the edited RGB
        # silhouette. Green means observed support covered by the edited view.
        residual = np.rint(0.30 * edited[name] + 0.70 * 255.0).astype(np.uint8)
        residual[support & edited_mask] = np.array([55, 205, 95], dtype=np.uint8)
        residual[support & ~edited_mask] = np.array([20, 205, 235], dtype=np.uint8)
        residual = _label(residual, name)
        residual_path = args.output_dir / f"edited_partial_support_{name}.png"
        Image.fromarray(residual).save(residual_path)
        residual_paths.append(residual_path)

        # Build a current-image residual vector field.  Each arrow starts at
        # the nearest edited foreground pixel and ends at an observed partial
        # pixel outside the silhouette.  Spatial thinning keeps it readable.
        distance, nearest = distance_transform_edt(
            ~edited_mask, return_distances=True, return_indices=True,
        )
        outside = np.argwhere(support & ~edited_mask)
        vector = np.rint(0.42 * edited[name] + 0.58 * 255.0).astype(np.uint8)
        chosen: list[tuple[int, int]] = []
        if len(outside):
            ordered = outside[np.argsort(distance[outside[:, 0], outside[:, 1]])[::-1]]
            occupied: set[tuple[int, int]] = set()
            cell = max(10, args.resolution // 24)
            for row, column in ordered:
                key = (int(row) // cell, int(column) // cell)
                if key in occupied or distance[row, column] < 1.0:
                    continue
                occupied.add(key)
                chosen.append((int(row), int(column)))
                source = (int(nearest[1, row, column]), int(nearest[0, row, column]))
                target = (int(column), int(row))
                cv2.circle(vector, source, 4, (230, 45, 45), -1, cv2.LINE_AA)
                cv2.circle(vector, target, 4, (25, 205, 225), -1, cv2.LINE_AA)
                cv2.arrowedLine(vector, source, target, (35, 35, 35), 5,
                                cv2.LINE_AA, tipLength=0.34)
                cv2.arrowedLine(vector, source, target, (250, 195, 25), 2,
                                cv2.LINE_AA, tipLength=0.34)
                if len(chosen) >= 64:
                    break
        vector = _label(vector, name)
        vector_path = args.output_dir / f"edited_to_partial_residual_{name}.png"
        Image.fromarray(vector).save(vector_path)
        vector_paths.append(vector_path)

        record = {"edited": _support_metrics(edited_mask, support)}
        if reference is not None:
            record["reference"] = _support_metrics(_object_mask(reference[name]), support)
            record["delta_coverage_3px"] = (
                record["edited"]["coverage_3px"] - record["reference"]["coverage_3px"]
            )
        record["displayed_residual_vectors"] = len(chosen)
        record["residual_arrow_direction"] = "current_edited_silhouette_to_partial_support"
        records[name] = record

    overlay_board = compose_grid(
        overlay_paths, args.output_dir / "edited_partial_overlay_front_side_back_right.png",
    )
    residual_board = compose_grid(
        residual_paths, args.output_dir / "edited_partial_support_front_side_back_right.png",
    )
    vector_board = compose_grid(
        vector_paths, args.output_dir / "edited_to_partial_residual_front_side_back_right.png",
    )
    report = {
        "method": "camera_locked_positive_partial_projection_check",
        "ground_truth_used": False,
        "missing_partial_pixels_are_unknown": True,
        "edited_board": str(args.edited_board.resolve()),
        "reference_board": str(args.reference_board.resolve()) if args.reference_board else None,
        "evidence_manifest": str(args.evidence_manifest.resolve()),
        "overlay_board": str(overlay_board.resolve()),
        "support_board": str(residual_board.resolve()),
        "residual_vector_board": str(vector_board.resolve()),
        "view_order": list(VIEW_ORDER),
        "views": records,
    }
    report_path = args.output_dir / "edited_partial_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
