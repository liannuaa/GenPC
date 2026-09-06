#!/usr/bin/env python3
"""Create a compact depth/semantic review board for isolated agent probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _tile(image_path: Path, title: str, *, size: int = 256) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 28), (20, 20, 20))
    tile.paste(image, (0, 28))
    ImageDraw.Draw(tile).text((6, 7), title, fill="white")
    return tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tiles: list[Image.Image] = []
    for sample in args.samples:
        camera = args.collection_root / sample / "inputs" / "camera" / sample
        tiles += [_tile(camera / "depth.png", f"{sample}: Camera-1 depth"),
                  _tile(camera / "img.png", f"{sample}: semantic observation")]
    width, height = tiles[0].size
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    board = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    for index, tile in enumerate(tiles):
        board.paste(tile, ((index % columns) * width, (index // columns) * height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
