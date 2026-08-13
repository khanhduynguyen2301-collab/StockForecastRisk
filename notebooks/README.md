# CLAUDE.md — Stock Forecasting & Risk Engine

Context for Claude Code working on this repository. Read this before making changes.

---

## 1. What this project is

A **stock price forecasting and risk engine** for an S&P 500 web dashboard. Two halves:

1. **Forecasting** — a pooled ML model predicting volatility from technical indicators.
2. **Risk** — a Merton jump-diffusion Monte Carlo producing price distributions, VaR, and CVaR.

This is **not a notebook experiment**. The end state is a backend engine a dashboard calls on demand, returning a forecast and a risk envelope as JSON within request latency. Notebooks are the research vehicle; `src/` modules are the deliverable.

**Dataset:** ~631k daily OHLCV rows, 502 tickers (503 minus `SW`, see §5), June 2021 – 2026.

---

## 2. The three findings that determine the design

Everything about the engine's architecture follows from three empirical results. **Do not silently reverse these** — they were expensive to establish and each is backed by a notebook.

| Question | Notebook | Answer |
|---|---|---|
| Do features predict 5-day **returns**? | 08 | **No.** Naive baseline beats Ridge and XGBoost on MAE in every fold; directional accuracy sits on the coin-flip line and its confidence interval includes 0.50. |
| Do features predict **volatility**? | 08b | **Yes.** Beats a persistence baseline in all 5 folds, R² ≈ 0.25–0.34, no leakage, driven by volatility-family features. |
| Are jumps needed, or is Gaussian enough? | 09 | **Jumps needed.** Gaussian simulations cannot reproduce the empirical kurtosis (5.49 on real data); Merton covers it comfortably. |
| Are the VaR numbers calibrated? | 11 | **Root cause found and fixed.** The Monte Carlo compounded daily steps to the horizon, which overstates the tail (cross-day mean reversion). Fitting directly on horizon returns calibrates — even a plain Gaussian. See §2b. |

### What this means for the engine

| Component | Source | Justified by |
|---|---|---|
| drift (μ) | risk-free rate (default), configurable | 08 — no return signal, so no ML steering |
| volatility (σ) | ML forecast (or historical) | 08b — beats persistence |
| jumps (λ per ticker, sizes pooled) | fitted from history | 09 — Gaussian misses the tails |

**The return null result is a finding, not a failure.** It is the documented justification for keeping drift simple. If someone later asks "why isn't the drift ML-steered?", notebook 08 is the answer.

---

## 2b. Calibration — ROOT CAUSE FOUND AND FIXED

**The engine's VaR overstated risk (breach rate 3–4% vs 5% expected) because it built horizon returns by compounding simulated daily steps.** That is now diagnosed to root cause and resolved.

### The diagnostic chain

1. Breach rate too low at every horizon (1d 3.0%, 5d 3.9%, 21d 3.4%), worst at 1-day.
2. Drift ruled out (wrong direction); overlapping-window artefact ruled out; sigma regime ruled out.
3. Jump term was the largest contributor, but removing jumps did not fully fix it, and **a Student-t diffusion failed identically to Merton** (3.84% vs 3.50%). Two different fat-tail mechanisms failing the same way pointed away from the tail model.
4. **Harness check settled it.** A historical VaR (empirical quantile) and a Gaussian fit — both run through the *same* breach-counting harness — calibrated cleanly (6.4% and 5.0%). The harness is sound.

### Root cause

The failing models fit parameters on **daily** returns, then simulate daily steps and **compound** to the horizon. Compounding i.i.d. daily draws assumes cross-day independence. Real returns have mild mean reversion at multi-day horizons, so actual H-day dispersion is *narrower* than independent compounding implies — a too-wide tail, too few breaches. This is also why 1-day was worst: no compounding, so the daily sigma overshoot (dispersion ratio 1.075) showed undiluted.

### The fix (verified)

**Fit and simulate at the horizon the VaR is reported for**, not by compounding daily. Models fit directly on overlapping H-day returns calibrate:

| Model (horizon-fit) | 5-day | 21-day |
|---|---|---|
| historical (empirical quantile) | p = 0.63 | p = 0.20 |
| gaussian on horizon returns | **p = 0.97** | **p = 1.00** |
| student-t on horizon returns | p = 0.70 | **p = 1.00** |

Notably, once fit at the horizon, **a plain Gaussian calibrates** — the daily fat tails largely wash out by 5–21 days (central-limit behaviour). The elaborate jump/t machinery is not needed at these horizons; it was compensating for a scaling error.

### Consequence for the engine design

- **Notebook 10's simulation is the wrong tool for horizon VaR as written.** It compounds daily steps. Either refit at the horizon, or (simpler and demonstrably calibrated) compute VaR from a horizon-return distribution directly.
- **Notebook 09's jump machinery is still correct for what it validates** (daily kurtosis) but is *not needed* for multi-day VaR. Do not delete it, but do not rely on it for the risk numbers.
- **The simplest calibrated path:** historical or Gaussian VaR on horizon returns, per ticker, refit point-in-time. This is what notebook 11's horizon-native section now uses.

### Still to do

- Re-run the horizon-native comparison on the full 50-ticker set at 5-day and 21-day (the section is built; run it).
- Decide whether to keep the Monte Carlo path engine at all for VaR, or use it only for path visualisation (fan charts) while computing risk numbers from horizon-fit distributions.

---

## 3. Repository state

### Production modules (`/mnt/user-data/outputs/`)

| File | Purpose | Status |
|---|---|---|
| `ingest.py` | yfinance ingestion, column normalisation, synthetic-padding cleaning | Tested |
| `indicators.py` | Feature engineering (42 features) | Tested |
| `schema.py` | `FEATURE_NAMES`, `TARGET_NAMES`, `RAW_COLUMNS`; SCHEMA_VERSION 2.1 | Tested |
| `test_schema.py` | 7 pytest tests — drift, leakage, point-in-time, numeric, target exclusion | All verified to fail correctly when bugs are injected |
| `loader.py` | Data loaders including `load_jump_diffusion_params()` | Tested |

### Notebooks (`notebooks/`)

| # | File | Role |
|---|---|---|
| 01 | `01_raw_data_audit.ipynb` | Data quality gate. Re-run on every data refresh. |
| 02 | `02_raw_market_visualization.ipynb` | Market intuition, ADF stationarity |
| 03 | `03_feature_statistics.ipynb` | Coverage, missingness, summary stats |
| 04 | `04_univariate_features.ipynb` | Distributions, normality (Jarque-Bera), per-feature stationarity |
| 05 | `05_bivariate_feature_target.ipynb` | ★ Signal check. Pearson vs Spearman, correlation stability over time |
| 06 | `06_multivariate_features.ipynb` | Redundancy, PCA, VIF, conditioning, redundancy stability |
| 07 | `07_outlier_analysis.ipynb` | 8 detectors compared; event-vs-error triage |
| 08 | `08_baseline_models.ipynb` | ★★ Return go/no-go (result: no signal) |
| 08b | `08b_volatility_baseline.ipynb` | ★★ Volatility go/no-go (result: signal) |
| 09 | `09_distribution_fitting.ipynb` | Merton parameter estimation |
| 10 | `10_monte_carlo_simulation.ipynb` | Path simulation, fan charts, VaR/CVaR |
| 11 | `11_var_backtest.ipynb` | ★★ Calibration check — Kupiec + Christoffersen |

### Run order

```
ingest.py → 01, 02 → indicators.py → 03–07 → 08, 08b → 09 → 10 → 11
```

- **01–02** read `data/raw/sp500_yfinance_daily.parquet`
- **03–08b** read `data/processed/sp500_features.parquet`
- **09** writes `data/processed/jump_diffusion_params.parquet`, consumed by **10**
- **11** refits parameters point-in-time; does not read that file

---

## 4. Key decisions and their reasoning

### Pooled model, not 503 per-ticker models
`ticker_id` and `sector_id` are features so the model learns cross-sectional patterns. Per-ticker models would have ~1,250 rows each — far too few.

### snake_case everywhere
Normalised at ingestion via `normalize_columns()` (regex `[^0-9a-z]+` → `_`). Canonical columns: `date, symbol, security, gics_sector, gics_sub_industry, open, high, low, close, adj_close, volume`.

**This migration broke things that were written earlier.** Notebook 08 referenced `"Date"`/`"Symbol"` and crashed with `KeyError` until fixed. `loader.py` had the same problem. If you find capitalised column references anywhere, they are stale.

### Both `adj_close` and `close` are kept (`auto_adjust=False`)
**Verified:** yfinance applies **split** adjustment to *both* columns. The gap between them is **dividends only**. NVDA's 10:1 split appears in both at ~120, not ~1200.

Consequence: a "split divergence" check between these columns can never fire. Notebook 01 §5 was rewritten to describe dividend adjustment and reframe the check as a general data-error tripwire. **Do not reintroduce split-detection logic based on the gap between these columns.**

### Drift is a deliberate choice, not the historical fit
`resolve_drift()` in notebook 10 supports `risk_free` (default, 4%), `zero`, `market` (8%), `historical`.

**Why:** σ converges in months; μ needs decades. A five-year drift estimate is mostly sample accident. Using per-ticker historical μ produced a 58.6% probability of loss over a quarter for a large-cap — implausible. Risk-free drift gives 51.4%.

**Empirical note:** VaR varies by only ~0.03 across drift modes, but `prob_loss` varies by ~0.10. **VaR and CVaR are robust to the drift choice; `prob_loss` and `median_return` are not.** Any UI showing the latter should state which drift mode produced them.

### Jump parameters are pooled, not per-ticker
`sigma` and `lambda` are per-ticker; `jump_mean` and `jump_std` are **pooled across all tickers**.

**Why:** ~26 flagged jumps per ticker is far too few to estimate a distribution. Notebook 09 includes a bootstrap demonstrating this: drawing per-ticker-sized samples from one identical population produces a `jump_std` spread *wider* than the spread actually observed across tickers. The apparent cross-sectional variation was sampling noise.

This was specified in the project blueprint and initially implemented incorrectly as per-ticker. **Do not revert.**

### Volatility estimation uses an iterative threshold estimator
A single σ over all returns is inflated by jumps. The estimator flags returns beyond 3σ, refits on the remainder, and repeats as the threshold tightens. Iteration matters — the first pass uses an inflated σ and misses smaller jumps.

### The jump compensator is not optional
In `simulate_paths()`, the drift includes `− λκ` where `κ = E[e^Y − 1]`. Without it, asymmetric jumps (equity `jump_mean` is negative) bias simulated prices low by ~9%. Notebook 10 has an `assert` verifying the mean terminal price matches theory within 2%. **Do not remove that assertion.**

---

## 5. Data quality: what was found and fixed

### `SW` — synthetic padding (RESOLVED)
355 rows with `volume ≤ 0`, all ticker `SW`. Diagnostics proved they were vendor-fabricated:
- `close == previous close` on 100% of them
- All four OHLC values identical on 100% of them
- 28.3% of `SW`'s 1,255 rows

**Fix:** `clean_synthetic_padding()` in `ingest.py`, wired into both export paths. Written as a **threshold rule**, not a hardcoded `SW` exclusion:
- Above 10% padded → drop the ticker (interleaved gaps contaminate every rolling window)
- Below 10% → NaN the OHLCV on those rows only, keeping the date so the series stays continuous

**Why dropping mattered:** 355 fabricated zero-return days would have suppressed `SW`'s volatility estimate badly — and per-ticker σ feeds the risk model directly.

### Everything else came back clean
- No price-sanity violations (`high < low`, `close > high`, zero prices)
- No duplicate `(date, symbol)` rows
- Trading-day coverage uniform: 502 tickers at exactly 50 missing business days (~47 expected from market holidays over 5 years); `FISV` at 51 from the FISV→FI rebrand
- Dividend adjustment sensible: top adjusted names are MO, LYB, VZ, T, F, KMI, OKE, VICI — a textbook high-yield roster; 93/503 non-payers

### Missing values are expected warm-up NaNs
`sma_200` ~16% (502 × 199), `return_20d` exact (502 × 21 = 10,542). Nothing at 100%. Requiring all features non-null leaves ~530k usable rows. **No action needed** — this is correct behaviour, not a bug.

---

## 6. Unresolved issues and known limitations

### Open
- **`days_to_next_earnings` may be all-NaN.** yfinance's `get_earnings_dates` was returning `KeyError` at one point. Notebooks guard against it by dropping entirely-missing features before `dropna`. Verify whether this column is populated before relying on it.
- **Portfolio VaR ignores correlation.** Notebook 10's portfolio section produces independent single-name figures and says so explicitly. Summing them understates joint downside. Genuine portfolio risk needs correlated shocks — a shared market factor or a copula over the diffusion terms.
- **Blueprint Phase 4b (steered drift) is obsolete.** It specifies ML probability warping the Monte Carlo drift (`μ_t = μ_0 + γ(p_t − 0.5)`). Notebook 08's null result supersedes it. The blueprint should be annotated so this is not built later from a plan that predates the evidence.
- **Notebook 10's forecast-σ path uses a stand-in.** It currently uses recent realised volatility to demonstrate the mechanism. Wiring in the actual trained 08b model is pending.

### Limitations to state, not fix
- **VaR backtest has low statistical power.** ~150 windows at a 5% expected rate means ~7 expected breaches. The Kupiec and Christoffersen tests detect gross miscalibration, not subtle bias. A passing result means "no evidence of a problem", not proof of correctness.
- **Backtest windows overlap.** Forecasts every 5 days for a 21-day horizon share data, so breaches are not fully independent. This inflates effective sample size and can make Christoffersen flag clustering that is partly an overlap artefact. Run non-overlapping if a result sits near significance.
- **Monte Carlo path count.** 10,000 paths gives ~1% error on the mean and 500 paths in the 5% tail — adequate for VaR95. **VaR99 needs more** (~100k for ~1,000 tail observations). Tail metrics converge much slower than the mean.

---

## 7. Working conventions

### Testing discipline
- **Execute notebooks end to end before trusting them.** This caught three real bugs: notebook 08's stale `"Date"` references, notebook 10's variable-ordering error (`resolve_drift` called before definition), and a scale mismatch in 08b's persistence baseline. None would have surfaced from reading the code.
- Notebooks import names from `schema.py` — **never hardcode feature lists**.
- Every notebook uses the same project-root bootstrap and reads the correct parquet.

### Point-in-time discipline
Non-negotiable throughout. Walk-forward splits with a purge gap in 08/08b; `shift(1)` before rolling windows in the rolling z-score (§07) so a row is never in its own baseline; parameters refit on past-only data in 11. **Any new feature or evaluation must preserve this.**

### Baselines must be on the same scale as the target
A scale mismatch in 08b's first draft compared an annualised volatility feature (~0.3) against a raw 5-day-std target (~0.018), making persistence look absurdly bad. The correct baseline is the target's own backward-looking twin. **Check this whenever adding a baseline.**

### Techniques earn their place or get documented as excluded
Notebook 07 runs 8 outlier detectors and explicitly records why others were left out (One-Class SVM: wrong scale; Elliptic Envelope: wrong distribution). This is deliberate — documenting considered-and-rejected is more useful than silently stacking methods.

### Macro (FRED) features were built, tested, and deliberately excluded
The full pipeline for market-wide macro factors exists — `macro.py` (FRED ingestion), the merge in `build_feature_dataset`, `MACRO_FEATURE_NAMES` / `MACRO_FEATURE_NAMES_FULL_HISTORY` in `schema.py`, `load_macro_factors` in the loader, and coverage checks in notebook 01. **The features are not used by the model, on purpose.**

- The ICE BofA credit spreads start only mid-2023 (notebook 01 flags this); the full-history subset excludes them.
- EDA (05/06) showed the remaining 11 are non-redundant with existing features but only weakly correlated with the target (VIX at 0.22).
- The A/B in 08b (same folds, base vs base+macro) showed macro **reduces** out-of-sample R²: Ridge −0.007, XGBoost −0.061. New but useless — the weak features gave the model more noise to overfit.

The plumbing is left in place (harmless — the merge is optional and skips when the file is absent) but **do not feed macro features to the volatility model.** This is a tested decision, not an oversight. Re-open only if a future data source provides per-ticker macro (e.g. options-implied vol), which targets 08b differently.

---

## 8. Next steps

### Immediate — nothing blocking

The research phase is complete. Calibration is solved (§2b: historical VaR on horizon returns), the volatility model is validated (08b), and the macro experiment is closed (excluded, see §7). No open modelling question blocks progress.

Optional refinements, only if wanted:
- **Tighten calibration.** Historical VaR passes marginally (p ≈ 0.064); the residual is skew, not kurtosis. A skew-t or Cornish-Fisher form is the known refinement. Not required — historical VaR is calibrated and simple.
- **Re-run 09 and 10 on real data** after the jump-pooling change, if the fan-chart outputs are still wanted for visualisation.

### Productionisation (blueprint Phase 5) — the main remaining work
Extract validated notebook logic into `src/` modules:
- `risk/historical_var.py` — **the calibrated VaR method** (point-in-time empirical horizon quantile). This is the risk number, not the Monte Carlo path engine. Add a regression test reproducing the ~4.6% breach rate.
- `training/train_volatility_model.py` — 08b's XGBoost, base features only (no macro), serialised and reloadable.
- `risk/jump_diffusion.py`, `risk/monte_carlo.py` — 09/10, kept for fan-chart *visualisation* only, not for the VaR number.
- `service/forecasting/registry.py` — model loading, target < 50ms.

This is mostly extraction; the logic is tested.

### Then: service and dashboard
API contract, caching/latency strategy, frontend.

### Deferred until justified
Hyperparameter tuning (must happen *inside* walk-forward, or it leaks), nested CV, feature selection, `arima_garch.py`, `lstm.py`, ensembling, trading-cost backtest, regime evaluation. **None of these are worth building until the volatility model is in production and its limits are known.**

---

## 9. Things to be careful about

- **Do not reverse the three findings in §2 without new evidence.** Especially: do not add ML-steered drift.
- **Do not remove the compensator assertion** in `simulate_paths()`.
- **Do not revert jump pooling** to per-ticker.
- **Do not reintroduce split-detection** based on the `close`/`adj_close` gap.
- **Do not hardcode `SW`** — the padding rule is general and will catch future cases.
- **Check for capitalised column names** if something fails with `KeyError`; they are stale from the pre-snake_case era.
- **Watch for scale mismatches** when comparing a model against a baseline.
- **Execute notebooks after editing them.** Static review has missed real bugs in this project more than once.
- **Never use overlapping windows for calibration testing.** They invalidate the Christoffersen independence test — the original 7/8 conditional-coverage failure was almost entirely this artefact.
- **Do not test calibration on 8 tickers.** It has no power; 50 tickers turned an apparent p = 0.85 pass into a p = 0.0002 failure on the same data.
- **Do not chase drift or horizon as the cause of the low breach rate.** Both were tested and ruled out; the sweep in notebook 11 documents the directions.
- **Do not read notebook 11's current pass as proof of calibration.** With 47 windows per ticker the tests have very little power; it means "no evidence of a problem". Widen the ticker set before production.
- **Do not tune calibration against synthetic data.** Synthetic random walks have no volatility regimes and gave the opposite sign on the EWMA result.
- **Do not feed macro (FRED) features to the volatility model.** They were built and A/B tested (§7) and reduced out-of-sample R² (Ridge −0.007, XGBoost −0.061). The plumbing existing in the repo is not a signal that the features should be used.
- **VaR comes from historical simulation on horizon returns, not the Monte Carlo path engine.** The path engine compounds daily steps, which overstates the tail (§2b). Notebooks 09/10 remain valid only for fan-chart visualisation.