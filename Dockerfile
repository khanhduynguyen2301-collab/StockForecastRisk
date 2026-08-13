# API container — serving only. Installs the lean serving requirements and runs the
# FastAPI app against the precomputed cache. It does NOT install training libraries
# and does NOT need the raw data panel or the vol model at runtime (see .dockerignore).
FROM python:3.11-slim

WORKDIR /app

# Install serving deps only (fast, small).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what serving needs: the engine source, the service, and the cache.
# .dockerignore excludes raw/processed data, notebooks, training, and the vol model.
COPY src/ ./src/
COPY service/ ./service/
COPY config/ ./config/
COPY models/serving_cache/ ./models/serving_cache/

EXPOSE 8000
CMD ["uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
