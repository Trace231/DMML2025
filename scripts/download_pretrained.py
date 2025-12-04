"""Download all pretrained weights used or supported by the project into a local folder.

Usage:
    python -m scripts.download_pretrained
    python scripts/download_pretrained.py --only torchvision
    python scripts/download_pretrained.py --only segformer

All weights are stored under: artifacts/pretrained/
"""

from __future__ import annotations

import argparse
import logging

from segmentation_benchmark.utils.pretrained import (
    download_all_torchvision_segmentation_weights,
    download_segformer_model,
    get_pretrained_root,
)


LOGGER = logging.getLogger("segmentation_benchmark.pretrained")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained model weights into a local folder")
    parser.add_argument(
        "--only",
        choices=["torchvision", "segformer"],
        nargs="*",
        help="Restrict download to a subset of providers (default: both)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    args = parse_args()
    setup_logging()

    root = get_pretrained_root()
    LOGGER.info("Using pretrained root directory: %s", root)

    only = set(args.only or ["torchvision", "segformer"])

    if "torchvision" in only:
        LOGGER.info("Downloading all torchvision segmentation weights...")
        paths = download_all_torchvision_segmentation_weights(force=args.force)
        LOGGER.info("Downloaded/verified %d torchvision weight files", len(paths))

    if "segformer" in only:
        # Currently we only have SegFormer-B0 in the codebase, but this helper is easy to extend.
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        LOGGER.info("Downloading SegFormer weights for '%s'...", model_name)
        download_segformer_model(model_name, force=args.force)
        LOGGER.info("SegFormer weights ready.")


if __name__ == "__main__":
    main()


