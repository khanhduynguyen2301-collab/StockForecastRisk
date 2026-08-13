"""Model registry — load the pooled model by version from the versioned store.

The shipping forecast model is the pooled VolatilityModel (gbm_boost). It is the
only model wired into serving; ARIMA/GARCH and LSTM were evaluated and set aside
(returns are not predictable at these horizons — see gbm_boost docstring and the
LSTM return test), so they are NOT part of the serving registry. Keeping them out
here is deliberate: the registry should reflect what actually ships.

"Versioned store" means: models live at models/forecast/<version>/ (each a directory
with model.json + metadata.json written by VolatilityModel.save). load_model reads a
named version, or the conventional `volatility_latest` pointer. The model's own
metadata carries schema_version, feature list, trained_on date, and row count, so a
loaded model is fully traceable.
"""
from __future__ import annotations

from pathlib import Path

from .gbm_boost import VolatilityModel

DEFAULT_MODEL_ROOT = Path("models/forecast")
LATEST_POINTER = "volatility_latest"


def load_model(version: str = LATEST_POINTER,
               model_root: str | Path = DEFAULT_MODEL_ROOT) -> VolatilityModel:
    """Load the pooled volatility model for a given version directory.

    `version` is a directory name under model_root (e.g. 'volatility_latest' or a
    dated 'volatility_xgb_2026-08-11'). The returned model carries its metadata
    (schema_version, feature_names, trained_on), so callers can verify provenance.
    """
    path = Path(model_root) / version
    if not path.exists():
        available = [p.name for p in Path(model_root).glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"No model version {version!r} under {model_root}. "
            f"Available: {available or '(none — run training first)'}"
        )
    return VolatilityModel.load(path)


def list_versions(model_root: str | Path = DEFAULT_MODEL_ROOT) -> list[str]:
    """All available model version directories in the store."""
    root = Path(model_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*") if p.is_dir())