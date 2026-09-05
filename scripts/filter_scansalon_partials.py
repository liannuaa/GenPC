#!/usr/bin/env python3
"""Create deterministic ScanSalon partial-point-cloud quality manifests.

The source data are never modified.  A sample is retained when its PLY vertex
count is at least ``--min-points``; both selected and rejected CSV manifests
retain the original metadata fields and add the measured point count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def ply_vertex_count(path: Path) -> int:
    """Read only the ASCII PLY header, including for binary point clouds."""
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip().split()
            if len(line) == 3 and line[:2] == ["element", "vertex"]:
                return int(line[2])
            if line == ["end_header"]:
                break
    raise ValueError(f"missing 'element vertex' header in {path}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/SCoDA/ScanSalon"))
    parser.add_argument("--min-points", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.min_points < 1:
        raise ValueError("--min-points must be positive")

    root = args.root.resolve()
    metadata_path = root / "metadata.csv"
    output_dir = (args.output_dir or root / "quality_filter").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = reader.fieldnames
        if source_fields is None or "pcd_filename" not in source_fields:
            raise ValueError(f"unexpected metadata columns in {metadata_path}")
        rows = list(reader)

    selected: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        relative_pcd = Path(row["pcd_filename"])
        point_count = ply_vertex_count(root / relative_pcd)
        enriched = {**row, "point_count": str(point_count)}
        (selected if point_count >= args.min_points else rejected).append(enriched)

    fields = [*source_fields, "point_count"]
    suffix = f"min{args.min_points}"
    write_csv(output_dir / f"metadata_{suffix}.csv", selected, fields)
    write_csv(output_dir / f"rejected_{suffix}.csv", rejected, fields)
    (output_dir / f"selected_{suffix}.txt").write_text(
        "".join(f"{row['pcd_filename']}\n" for row in selected), encoding="utf-8"
    )
    summary = (
        f"minimum_points={args.min_points}\n"
        f"total={len(rows)}\n"
        f"selected={len(selected)}\n"
        f"rejected={len(rejected)}\n"
    )
    (output_dir / f"summary_{suffix}.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"selected_manifest={output_dir / f'metadata_{suffix}.csv'}")
    print(f"rejected_manifest={output_dir / f'rejected_{suffix}.csv'}")


if __name__ == "__main__":
    main()
