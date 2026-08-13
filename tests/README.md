# Test suite

95 tests guarding the invariants that matter — the honest contract, calibration
discipline, leakage guards, risk math, and single-source-of-truth agreements. Every
test targets a real invariant or a bug actually hit during development, not coverage
for its own sake.

## Run

```bash
pytest                 # whole suite
pytest tests/api       # one area
pytest -q              # quiet
```

Needs the serving dependencies plus `pytest` (`pip install -r requirements.txt pytest`).
No training libraries (xgboost/sklearn) required — those modules are tested in the
training environment, not here.

## Layout

- `conftest.py` — path setup + shared fixtures (`serving_cache`, `returns_series`,
  `api_client`).
- `api/` — the public surface. **The honest contract is enforced here.**
  - `test_contract.py` — no directional fields, horizon 422 guard, central estimate =
    risk median, disclosure/window match, 404s, disclaimer.
  - `test_calibration_guard.py` — 503 refusal when engine window ≠ disclosed window.
  - `test_endpoints.py` — `/tickers`, `/health`, `/`, prediction logging, ticker
    case/whitespace, service exception distinctions, schema-level guards.
  - `test_main.py` — app assembly: metadata, routing, CORS, docs, read-only surface.
- `risk/` — the risk math.
  - `test_filtered_var.py` — VaR/CVaR coherence, per-horizon windows, no look-ahead.
  - `test_historical_var.py` — unconditional VaR (used by the backtest comparison).
  - `test_monte_carlo.py` — fan-chart simulator; the compensator/drift gate.
- `evaluation/` — `test_splits_baselines.py`: the walk-forward leakage guard + baselines.
- `forecasting/` — `test_jump_diffusion.py`: jump fit + the pooling invariant.
- `features/` — `test_schema.py` (vol feature-list exclusions), `test_feature_parity.py`
  (train/serve divergence guard), `test_data_pipeline.py` (symbol normalization, FINRA
  parsing, regimes, IO).
- `test_config.py` — config consistency and config↔engine agreement.

## Not covered here (by design)

- `gbm_boost.py` (VolatilityModel) — needs xgboost; tested in the training env.
- `quantile_volatility.py` — real logic, but not in the v1 serving path.
- `indicators.py` / `loader.py` / `fundamentals.py` / `macro.py` — offline-pipeline
  modules that need realistic fixture frames; genuine future coverage.
- `lstm.py` / `arima_garch.py` — the abandoned approaches; not in the product.
