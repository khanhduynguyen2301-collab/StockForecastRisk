"""StockForecastRisk API — the HTTP serving surface for the honest risk engine.

Serves precomputed, calibrated risk estimates (filtered historical simulation) for
S&P 500 tickers at validated horizons (5/10/21-day). It deliberately does NOT expose
directional return forecasts: they do not beat a no-change baseline and are held out.

Reads the same serving cache as the Streamlit app, so both surfaces return identical
numbers. Run precompute_serving_cache.py to (re)build that cache.

Run:  uvicorn service.api.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="StockForecastRisk API",
    version="1.0.0",
    description=(
        "Honest tail-risk and volatility estimates for S&P 500 stocks. "
        "A modeled range of outcomes, not investment advice. No directional "
        "prediction is provided."
    ),
)

# CORS: allow a frontend to call the API. Tighten allow_origins to your frontend's
# domain before any real deployment; "*" is fine only for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")


@app.get("/")
def root() -> dict:
    return {
        "service": "StockForecastRisk API",
        "version": "1.0.0",
        "docs": "/docs",
        "disclaimer": "For research and educational use only. Not investment advice.",
    }