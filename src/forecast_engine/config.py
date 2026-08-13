"""Typed settings loader.

Tunables live in config/config.yaml (committed, read by BOTH training and serving).
Secrets live in .env (never committed). This module loads the YAML into typed
pydantic Settings and layers .env / environment secrets on top, so there is exactly
one place each kind of setting is authored and a single typed object everything reads.

Precedence (highest first): real environment variables > .env > config.yaml > field
defaults. That lets a deployment override any tunable via an env var without editing
the YAML, while the YAML remains the canonical source for the committed tunables.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# config/config.yaml relative to the repo root (this file is at
# src/forecast_engine/config.py, so the repo root is two parents up).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_YAML = _REPO_ROOT / "config" / "config.yaml"


class _YamlSource(PydanticBaseSettingsSource):
    """A settings source that reads tunables from config/config.yaml."""

    def get_field_value(self, field, field_name):  # required by the ABC; unused
        return None, field_name, False

    def __call__(self) -> dict:
        if not _CONFIG_YAML.exists():
            return {}
        with _CONFIG_YAML.open("r", encoding="utf-8") as f:
            return dict(yaml.safe_load(f) or {})


class Settings(BaseSettings):
    # --- service ---
    env: str = "development"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --- horizons ---
    horizon: int = 21
    offered_horizons: list[int] = [5, 10, 21]

    # --- risk model ---
    confidence_level: float = 0.95
    n_paths: int = 1000
    horizon_vol_window: dict[int, int] = {5: 21, 10: 21, 21: 42}
    horizon_disclosure: dict[int, dict[str, str]] = {
        5:  {"breach": "5.4%", "crisis": "~7%",   "window": "21"},
        10: {"breach": "5.5%", "crisis": "~7.5%", "window": "21"},
        21: {"breach": "5.8%", "crisis": "~8%",   "window": "42"},
    }

    # --- paths ---
    data_sample_dir: Path = Path("data/sample")
    model_dir: Path = Path("models")
    serving_cache_path: Path = Path("models/serving_cache/serving_cache.parquet")
    forecast_model_dir: Path = Path("models/forecast/volatility_latest")
    jump_params_path: Path = Path("models/risk_params/jump_diffusion.parquet")
    prediction_log_path: Path = Path("logs/predictions.jsonl")

    # --- secrets (from .env only; never in YAML) ---
    redis_url: str = "redis://redis:6379"
    finra_client_id: str | None = None
    finra_client_secret: str | None = None
    edgar_user_agent: str | None = None

    def window_for(self, horizon: int) -> int:
        """The validated conditioning window for a horizon (default 21)."""
        return self.horizon_vol_window.get(horizon, 21)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings,
        dotenv_settings, file_secret_settings,
    ):
        """Precedence: env > .env > config.yaml > defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()