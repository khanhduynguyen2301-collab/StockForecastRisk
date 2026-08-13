.PHONY: install install-train train fit-jump precompute calibrate serve app \
        docker-up docker-down test lint format

# --- setup ------------------------------------------------------------------
install:            ## serving deps only
	pip install -r requirements.txt

install-train:      ## serving + training/pipeline deps
	pip install -r requirements.txt -r requirements-train.txt

# --- offline pipeline (the "slow path"; run after a data refresh) -----------
train:              ## train + validate the volatility model (gates on persistence baseline)
	python -m training.pipeline.train_forecast_models

fit-jump:           ## fit per-ticker jump-diffusion params (for the illustrative fan)
	python -m training.pipeline.fit_jump_diffusion_params

precompute:         ## build the serving cache the API + app read (run last)
	python -m training.pipeline.precompute_serving_cache

calibrate:          ## disclosure breach rates per offered horizon
	python -m training.validation.sweep_horizon_disclosure

validate:           ## per-horizon window calibration sweep
	python -m training.validation.validate_filtered_var

# Full offline rebuild in dependency order.
build-all: train fit-jump precompute
	@echo "Offline artifacts rebuilt. Serving cache is ready."

# --- serving (the "fast path") ----------------------------------------------
serve:              ## run the API locally (http://localhost:8000/docs)
	uvicorn service.api.main:app --reload

app:                ## run the Streamlit dashboard locally (http://localhost:8501)
	streamlit run apps/streamlit/app.py

# --- docker -----------------------------------------------------------------
docker-up:          ## build + run both API and Streamlit containers
	docker compose up --build

docker-down:
	docker compose down

# --- quality ----------------------------------------------------------------
test:
	pytest

lint:
	black --check src service training && isort --check src service training

format:
	black src service training && isort src service training
