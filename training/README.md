# training/

Offline jobs — the "slow path" of the offline/online split. Everything here writes
artifacts (models, params, the serving cache) or produces validation evidence. None
of it runs on the serving request path. Grouped by how often you actually run it.

## pipeline/ — the recurring rebuild (run on every data refresh)

Run in dependency order; `make build-all` chains all three:

1. `train_forecast_models.py` — fits + validates the pooled volatility model
   (gates on the persistence baseline). → `models/forecast/volatility_latest/`
2. `fit_jump_diffusion_params.py` — fits per-ticker σ/λ + pooled jump params, writes
   a versioned manifest. → `models/risk_params/jump_diffusion.parquet` (+ manifest)
3. `precompute_serving_cache.py` — loads a model version via the registry, runs the
   engine for every ticker × horizon, stamps model + param versions into each row.
   → `models/serving_cache/serving_cache.parquet` (what the API and app read)

```bash
make build-all          # or: python -m training.pipeline.train_forecast_models  (etc.)
```

## validation/ — calibration evidence (re-run when the method or data changes)

Not part of the routine rebuild; these produce the numbers behind the disclosed
calibration claims. Re-run them when you change the VaR method or refresh the panel,
to confirm the disclosed breach rates still hold.

- `validate_filtered_var.py` — per-horizon conditioning-window sweep (picks the window).
- `sweep_horizon_disclosure.py` — pooled breach rate per offered horizon (the numbers
  the app/API disclose).
- `run_kupiec_disclosure.py` — aggregate Kupiec breach disclosure.
- `run_var_calibration.py` — calibration check (uses `backtest.py`).
- `backtest.py` — the point-in-time breach backtest harness.

```bash
make validate           # or: python -m training.validation.sweep_horizon_disclosure  (etc.)
```

## diagnostics/ — one-off investigation tools (rarely re-run)

- `diagnose_var_by_year.py` — per-year breach breakdown; this is the tool that
  diagnosed the tail-clustering that drove the filtered-VaR fix.
- `smoke_test_orchestrator.py` — end-to-end wiring sanity check on real data.

```bash
python -m training.diagnostics.diagnose_var_by_year
```

## Run order note

All jobs use absolute imports (`from src.forecast_engine...`) and are invoked as
modules from the repo root (`python -m training.<group>.<script>`). Run `pipeline/`
before `validation/` (validation reads the artifacts the pipeline produces).
