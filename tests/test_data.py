from pathlib import Path

import numpy as np
from PIL import Image

from segmentation_benchmark.data import CrackForestDataset, create_dataloaders


def _make_sample(root: Path, name: str = "sample") -> None:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    mask = Image.fromarray((np.random.rand(128, 128) > 0.5).astype("uint8") * 255)
    image.save(images_dir / f"{name}.jpg")
    mask.save(masks_dir / f"{name}.jpg")


def test_crackforest_dataset_custom_root(tmp_path):
    root = tmp_path / "cf"
    for idx in range(3):
        _make_sample(root, name=f"sample_{idx}")
    dataset = CrackForestDataset(root=root, image_size=64, augment=False)
    item = dataset[0]
    assert item["image"].shape[-2:] == (64, 64)
    assert item["mask"].shape[-2:] == (64, 64)


def test_create_dataloaders_without_download(tmp_path):
    root = tmp_path / "cf"
    for idx in range(6):
        _make_sample(root, name=f"sample_{idx}")
    config = {
        "root": str(root),
        "download": False,
        "image_size": 64,
        "batch_size": 2,
        "num_workers": 0,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "augment": False,
    }
    loader_dict = create_dataloaders(config)
    assert "train_loader" in loader_dict
    batch = next(iter(loader_dict["test_loader"]))
    assert batch["image"].shape[0] <= 2
```