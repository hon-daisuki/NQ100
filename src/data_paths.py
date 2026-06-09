from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DrivePaths:
    data_dir: str
    features_csv: str = "USTEC_features_all.csv"
    reports_dir: str | None = None
    models_dir: str | None = None

    @property
    def features_path(self) -> str:
        return str(Path(self.data_dir) / self.features_csv)


DEFAULT_COLAB_PATHS = DrivePaths(
    data_dir="/content/drive/MyDrive/CFD機械学習/backtest_ready",
)
