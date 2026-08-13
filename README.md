# StockForecastRisk

A stock forecasting and risk modeling project with a thin FastAPI service wrapper and offline training workflows.

## Project Layout

- `pyproject.toml` - dependency and tool configuration
- `Makefile` - common tasks for train / serve / drift-check / test
- `Dockerfile` - container image for the FastAPI service
- `docker-compose.yml` - local stack with Redis
- `.env.example` - required environment variables
- `config/` - runtime config and logging settings
- `src/forecast_engine/` - installable package with core forecasting and risk logic
- `service/api/` - FastAPI deployment wrapper
- `training/` - offline model training and evaluation jobs
- `apps/streamlit/` - research UI
- `data/` - input data directories and committed sample data
- `models/` - trained model artifacts (gitignored)
- `logs/` - runtime logging and prediction history (gitignored)
- `tests/` - project tests mirroring package layout

## Setup

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the API locally:
   ```powershell
   uvicorn service.api.main:app --reload
   ```

4. Run tests:
   ```powershell
   pytest
   ```

## Local Container

```powershell
docker compose up --build
```
