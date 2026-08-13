"""StockForecastRisk — v1 risk dashboard (REAL engine).

Wires the honest engine into the UI shell. Two sources, clearly separated:

  - NUMBERS (predicted volatility, VaR, CVaR, percentiles) come from the calibrated
    engine: orchestrator.build_response() -> filtered_historical_var. These are the
    risk statements.
  - PICTURE (fan chart, terminal histogram) comes from monte_carlo.simulate_paths,
    which is illustrative ONLY (it overstates the compounded tail — see its docstring).
    Labelled as such so the smooth visual is never mistaken for the calibrated numbers.

The forecast is VOLATILITY, not direction — no up/down arrow, no probability_up. The
central estimate is the risk median. The calibration_note (normal-conditions vs severe
stress) is surfaced in the validation panel.

Requires (run these first):
  - training/train_forecast_models.py     -> models/forecast/volatility_latest
  - training/fit_jump_diffusion_params.py -> models/risk_params/jump_diffusion.parquet

Run:  streamlit run apps/streamlit/app.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path.cwd().resolve()
for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (parent / "src").is_dir():
        PROJECT_ROOT = parent
        break
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_engine.data.loader import load_processed_features
from src.forecast_engine.forecasting.gbm_boost import VolatilityModel
from src.forecast_engine.orchestrator import build_response
from src.forecast_engine.risk import monte_carlo as mc

MODEL_DIR = PROJECT_ROOT / "models" / "forecast" / "volatility_latest"
JUMP_PARAMS = PROJECT_ROOT / "models" / "risk_params" / "jump_diffusion.parquet"

# Per-horizon disclosed breach rates (from training/sweep_horizon_disclosure.py,
# filtered historical simulation, VOL_WINDOW=21, 2010-2026, 497 tickers). Update
# these if the panel or method changes. The note shows the figure for the horizon
# on screen so the disclosure always matches the number displayed. 60-day is not
# offered because it does not calibrate at any conditioning window.
# Per-horizon disclosed breach rates AND the conditioning window each was measured
# at (from training/sweep_horizon_disclosure.py, filtered historical simulation,
# 2010-2026, 497 tickers). Each horizon ships with its validated window: 5/10-day
# at w21, 21-day at w42. The `window` here MUST match the engine's vol_window for
# that horizon, or the disclosed figure describes a different computation than the
# one shown — render() asserts this and warns if they ever diverge. 60-day is not
# offered (does not calibrate at any window).
HORIZON_DISCLOSURE = {
    5:  {"breach": "5.4%", "crisis": "~7%",   "window": 21},
    10: {"breach": "5.5%", "crisis": "~7.5%", "window": 21},
    21: {"breach": "5.8%", "crisis": "~8%",   "window": 42},
}

INK, SLATE, PAPER, MUTED = "#1a1d24", "#2b303b", "#f6f5f1", "#8a8f9a"
BAND_OUTER, BAND_INNER, MEDIAN, LOSS = "#c9cdd6", "#9aa2b1", "#3a4658", "#b5563f"

st.set_page_config(page_title="Risk Engine — v1", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"""
<style>
  .stApp {{ background: {PAPER}; }}
  h1,h2,h3,h4 {{ color:{INK}; font-family:'Georgia',serif; letter-spacing:-0.01em; }}
  .eyebrow {{ color:{MUTED}; font-size:0.72rem; text-transform:uppercase;
             letter-spacing:0.18em; font-weight:600; }}
  .bignum {{ font-family:'Georgia',serif; font-size:2.4rem; color:{INK}; line-height:1.05; }}
  .caption {{ color:{MUTED}; font-size:0.82rem; }}
  .metricbox {{ background:white; border:1px solid #e4e2db; border-radius:6px; padding:14px 16px; }}
  .lossnum {{ color:{LOSS}; font-family:'Georgia',serif; font-size:1.5rem; }}
  .framing {{ color:{SLATE}; font-size:0.9rem; font-style:italic;
             border-left:3px solid {BAND_INNER}; padding-left:12px; margin:6px 0 2px; }}
  .illus {{ color:{MUTED}; font-size:0.72rem; font-style:italic; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    model = VolatilityModel.load(MODEL_DIR)
    jump = pd.read_parquet(JUMP_PARAMS).set_index("symbol")
    return model, jump


@st.cache_resource
def load_panel():
    df = load_processed_features()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def daily_log_returns(frame: pd.DataFrame) -> np.ndarray:
    c = frame.sort_values("date")["adj_close"].astype(float)
    return np.log(c / c.shift(1)).dropna().to_numpy()


def simulate_for_visual(jump_row: pd.Series, last_price: float, horizon: int):
    """Illustrative Monte Carlo paths for the fan/histogram — NOT a risk statement."""
    params = mc.JumpDiffusionParams(
        mu_annual=0.0,  # risk-neutral-ish drift for the visual; not a return forecast
        sigma_annual=float(jump_row["sigma_annual"]),
        lambda_per_year=float(jump_row["lambda_per_year"]),
        jump_mean=float(jump_row["jump_mean"]),
        jump_std=float(jump_row["jump_std"]),
    )
    paths = mc.simulate_paths(last_price, params, horizon_days=horizon, n_paths=8000)
    fan = mc.fan_chart_percentiles(paths)
    terminal = mc.terminal_distribution(paths, last_price)
    return fan, terminal


def fan_chart(fan: dict, horizon: int) -> go.Figure:
    d = list(range(1, horizon + 1))
    to_ret = lambda arr: ((np.asarray(arr) / arr[0]) - 1.0)  # start-relative return
    p5, p25, p50, p75, p95 = (fan[f"p{p}"] for p in (5, 25, 50, 75, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d + d[::-1], y=list(to_ret(p95)) + list(to_ret(p5))[::-1],
                             fill="toself", fillcolor=BAND_OUTER, line=dict(width=0),
                             name="5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d + d[::-1], y=list(to_ret(p75)) + list(to_ret(p25))[::-1],
                             fill="toself", fillcolor=BAND_INNER, line=dict(width=0),
                             name="25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=to_ret(p50), line=dict(color=MEDIAN, width=2), name="median"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                      showlegend=True, legend=dict(orientation="h", y=-0.18),
                      xaxis_title="trading days ahead", yaxis_title="cumulative return")
    fig.update_yaxes(tickformat=".1%")
    return fig


def hist_chart(terminal: np.ndarray, var_95: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=terminal, nbinsx=60, marker_color=BAND_INNER, opacity=0.85))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    if var_95 is not None:
        fig.add_vline(x=var_95, line=dict(color=LOSS, width=2),
                      annotation_text="VaR 95% (calibrated)", annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                      showlegend=False, xaxis_title="return at horizon", bargap=0.02)
    fig.update_xaxes(tickformat=".0%")
    return fig


def render(ticker: str, horizon: int, model, jump, panel):
    feats = panel[panel["symbol"] == ticker].copy()
    if feats.empty:
        st.warning(f"No data for {ticker}.")
        return
    returns = daily_log_returns(feats)
    resp = build_response(
        ticker=ticker, as_of=feats["date"].max().date(),
        volatility_model=model, ticker_features=feats,
        daily_log_returns=returns, horizon_days=horizon, risk_params_version="v1",
    )
    fc, risk = resp["forecast"], resp["risk"]

    st.markdown(f'<div class="eyebrow">{ticker} · {horizon}-day horizon</div>',
                unsafe_allow_html=True)

    # central estimate = risk median (or predicted vol if risk unavailable)
    if risk is not None:
        mid = risk["percentiles"]["p50"]
        st.markdown(f'<div class="bignum">{mid:+.2%}</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption">central estimate · risk-model median</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="framing">Modeled range of outcomes, not a prediction of '
                'direction.</div>', unsafe_allow_html=True)

    # predicted volatility (the validated forecast)
    pv = fc.get("predicted_volatility")
    if pv is not None:
        st.markdown(f'<div class="caption">forecast volatility (daily): '
                    f'<b>{pv:.4f}</b></div>', unsafe_allow_html=True)

    # illustrative visuals from Monte Carlo
    last_price = float(feats.sort_values("date")["adj_close"].iloc[-1])
    var_for_line = risk["var_95"] if risk else None
    try:
        fan, terminal = simulate_for_visual(jump.loc[ticker], last_price, horizon)
        st.plotly_chart(fan_chart(fan, horizon), width="stretch",
                        key=f"fan_{ticker}_{horizon}")
        st.markdown('<div class="illus">Fan &amp; distribution are illustrative Monte-Carlo '
                    'paths. The VaR/CVaR numbers below come from the calibrated engine, not '
                    'from these paths.</div>', unsafe_allow_html=True)
        st.plotly_chart(hist_chart(terminal, var_for_line), width="stretch",
                        key=f"hist_{ticker}_{horizon}")
    except KeyError:
        st.info(f"No jump-diffusion params for {ticker}; charts unavailable.")

    # calibrated risk numbers
    if risk is not None:
        c1, c2, c3 = st.columns(3)
        for col, label, val in [(c1, "VaR 95%", risk["var_95"]),
                                 (c2, "CVaR 95%", risk["cvar_95"]),
                                 (c3, "median", risk["percentiles"]["p50"])]:
            cls = "lossnum" if "VaR" in label or "CVaR" in label else "bignum"
            style = "" if cls == "lossnum" else "font-size:1.5rem"
            col.markdown(f'<div class="metricbox"><div class="caption">{label}</div>'
                         f'<div class="{cls}" style="{style}">{val:+.2%}</div></div>',
                         unsafe_allow_html=True)

        with st.expander("Calibration & validation detail"):
            note = risk.get("calibration_note", "")
            st.markdown(f"**Risk method:** {risk['model']} · "
                        f"{risk['n_samples']:,} horizon samples")
            if note:
                st.markdown(f"*{note}*")
            disc = HORIZON_DISCLOSURE.get(horizon)
            if disc:
                # Safety: the disclosed figure was measured at disc["window"]; the
                # engine must have used the same window, or the note describes a
                # different computation than the VaR shown.
                engine_window = risk.get("vol_window")
                if engine_window is not None and engine_window != disc["window"]:
                    st.warning(
                        f"Calibration mismatch: disclosed figure measured at "
                        f"window {disc['window']}, engine used {engine_window}. "
                        f"Disclosure suppressed.")
                else:
                    st.markdown(
                        f"- Pooled {horizon}-day 95% VaR breach rate **{disc['breach']}** "
                        f"vs 5% target (2010–2026, 497 tickers, point-in-time, "
                        f"vol-window {disc['window']}).\n"
                        f"- Calibrated in normal conditions; modestly understates severe "
                        f"stress (crisis-year breach {disc['crisis']}).\n"
                        "- Directional return forecast is **not shown**: it does not beat a "
                        "no-change baseline and is held server-side.")
            else:
                st.markdown(
                    "- Calibrated in normal conditions; may modestly understate severe "
                    "stress.\n"
                    "- Directional return forecast is **not shown**: it does not beat a "
                    "no-change baseline and is held server-side.")
    else:
        st.info("Insufficient history to estimate risk for this ticker.")


# ---- layout ----
st.markdown('<div class="eyebrow">StockForecastRisk</div>', unsafe_allow_html=True)
st.title("Risk & outcome-range engine")
st.markdown('<div class="caption">A modeled range of outcomes with tail risk — not a '
            'directional prediction. Volatility/risk engine, v1.</div>',
            unsafe_allow_html=True)

try:
    model, jump = load_engine()
    panel = load_panel()
except FileNotFoundError as e:
    st.error(f"Engine artifacts missing — run training + jump-param fitting first.\n\n{e}")
    st.stop()

tickers = sorted(panel["symbol"].unique())
st.divider()
c1, c2, c3 = st.columns([2, 1, 1])
compare = c1.toggle("Compare two tickers", value=False)
horizon = c2.selectbox("Horizon (trading days)", [5, 10, 21], index=2)
c3.markdown('<div class="caption" style="padding-top:8px">real engine</div>',
            unsafe_allow_html=True)
st.divider()

if compare:
    lc, rc = st.columns(2)
    with lc:
        render(st.selectbox("Ticker A", tickers, index=0, key="tA"), horizon, model, jump, panel)
    with rc:
        render(st.selectbox("Ticker B", tickers, index=min(1, len(tickers) - 1), key="tB"),
               horizon, model, jump, panel)
else:
    render(st.selectbox("Ticker", tickers, index=0, key="tS"), horizon, model, jump, panel)

st.divider()
st.caption("Numbers: calibrated engine (filtered historical simulation). "
           "Fan/histogram: illustrative Monte-Carlo, not a risk statement.")