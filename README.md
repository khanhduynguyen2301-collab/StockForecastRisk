# StockForecastRisk

> A calibrated volatility and tail-risk engine for S&P 500 equities — built to report only what the evidence supports.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

StockForecastRisk forecasts **volatility** and reports **calibrated Value-at-Risk (VaR)
and Conditional VaR** for ~500 S&P 500 stocks. It presents a modeled *range of
outcomes* with explicit tail-risk estimates — not a directional price prediction, and
not investment advice.

Its defining design principle is **epistemic honesty**: the engine ships only signals
that survive rigorous out-of-sample validation, and it is explicit about the limits of
those it ships.

---

## Why this project is different

Most retail "stock prediction" tools claim to forecast direction or price. This one
deliberately does not — because the evidence doesn't support it.

Return- and direction-prediction models (pooled gradient-boosted trees, an LSTM, and a
cross-sectional ranking signal) were built and tested against a naive no-change
baseline. **None beat the baseline out of sample** (e.g. an LSTM on daily returns
scored 50.5% directional accuracy — a coin flip). Rather than ship a model that looks
impressive but adds no skill, those components were excluded from the product.

What *did* validate — a volatility forecast and a calibrated risk model — is what
ships. The API contract has no `predicted_direction`, `probability_up`, or
`predicted_return` field, and this is **enforced at the type level**: the dishonest
response is not representable.

---

## What it does

| Component | Approach | Status |
|---|---|---|
| **Volatility forecast** | One pooled XGBoost regressor across all tickers, with `ticker_id` / `sector_id` as features | Beats persistence baseline 5/5 walk-forward folds (~15% MAE reduction) |
| **Tail risk (VaR / CVaR)** | Filtered historical simulation — returns standardized by trailing volatility, then rescaled to the current regime | Calibrated per horizon; breach rates disclosed |
| **Outcome fan chart** | Merton jump-diffusion Monte Carlo | Illustrative only — never the source of the risk numbers |
| **Return / direction** | Pooled GBM, LSTM, cross-sectional ranking | Tested, did not beat baseline — **deliberately not shipped** |

### Calibration, disclosed honestly

Horizons of **5, 10, and 21 trading days** are offered, each calibrated point-in-time
over 2010–2026 across ~497 tickers:

| Horizon | Conditioning window | Pooled 95% VaR breach rate (target: 5%) |
|---|---|---|
| 5-day | 21-day | 5.4% |
| 10-day | 21-day | 5.5% |
| 21-day | 42-day | 5.8% |

The engine is well-calibrated in normal conditions and **modestly understates risk in
severe stress** (crisis-year breach ~7–8%) — a limitation disclosed in every response
rather than hidden. A 60-day horizon was evaluated and **excluded** because it does not
calibrate at any conditioning window.

---

## Architecture

Core logic lives once in `src/forecast_engine/`; every other layer is a thin consumer
that imports it. This makes train/serve feature parity a **structural guarantee**, not
a convention — there is a single implementation of feature engineering, and both
training and inference use it.

```
src/forecast_engine/     Core: features, models, risk, orchestration (single source of truth)
service/api/             FastAPI service — reads the serving cache, honest typed contract
apps/streamlit/          Dashboard — reads the same cache, identical numbers
training/
  pipeline/              Recurring rebuild: train → fit params → precompute cache
  validation/            Calibration evidence (breach-rate sweeps, backtests)
  diagnostics/           One-off investigation tools
config/config.yaml       Tunables, read identically by training and serving
models/                  Versioned, provenance-stamped artifacts (output)
tests/                   96 tests: honest contract, calibration, leakage guards, risk math
```

**Offline / online split.** Training fits models and precomputes a compact serving
cache; the API and dashboard only *read* that cache. Serving is therefore a cache
lookup, not on-request inference — fast, light, and identical across both surfaces.
Every served number is provenance-stamped with the model and parameter versions that
produced it.

---

## Quickstart

```bash
# Install (serving + training dependencies)
pip install -r requirements.txt -r requirements-train.txt

# Build the artifacts (in order)
python -m training.pipeline.train_forecast_models       # fit + validate the volatility model
python -m training.pipeline.fit_jump_diffusion_params   # fit jump-diffusion parameters
python -m training.pipeline.precompute_serving_cache    # build the serving cache

# Serve
uvicorn service.api.main:app --reload                   # API  → http://localhost:8000/docs
streamlit run apps/streamlit/app.py                     # App  → http://localhost:8501
```

**No dataset?** `src/forecast_engine/data/synthetic.py` generates a schema-matching
panel so the full pipeline runs end-to-end without private data (for demonstration and
testing — the synthetic series carry no real signal).

### API

```bash
GET /v1/forecast/{ticker}?horizon={5|10|21}
```

Returns the volatility forecast, calibrated VaR/CVaR with per-horizon percentiles, a
calibration note, and provenance versions. Interactive docs at `/docs`.

### Docker
