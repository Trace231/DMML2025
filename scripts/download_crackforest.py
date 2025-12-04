"""Utility script to download the CrackForest dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from segmentation_benchmark.data.crackforest import ensure_crackforest_dataset
from segmentation_benchmark.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the CrackForest dataset")
    parser.add_argument("--target", type=Path, default=None, help="Target directory for the dataset")
    parser.add_argument("--no-download", action="store_true", help="Fail if dataset missing instead of downloading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root()
    target = args.target or paths.data_dir / "crackforest"
    ensure_crackforest_dataset(target, download=not args.no_download)
    print(f"CrackForest dataset is ready at {target}")


if __name__ == "__main__":
    main()
