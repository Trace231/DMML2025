"""Project path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    """Convenience wrapper that centralises frequently used project folders."""

    root: Path
    data_dir: Path
    artifacts_dir: Path
    # Folder for all pretrained model weights (torchvision, HuggingFace, etc.)
    pretrained_dir: Path
    reports_dir: Path
    configs_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Optional[Path] = None) -> "ProjectPaths":
        """Instantiate using the provided project root or auto-detect it."""
        if root is None:
            # Auto-detect by hopping four levels up from this file: utils -> package -> src -> project root.
            root = Path(__file__).resolve().parents[3]
        root = root.resolve()
        return cls(
            root=root,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
            pretrained_dir=root / "artifacts" / "pretrained",
            reports_dir=root / "reports",
            configs_dir=root / "configs",
            logs_dir=root / "logs",
        )

    def ensure(self) -> None:
        """Create all managed directories if they are missing."""
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.pretrained_dir,
            self.reports_dir,
            self.configs_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

