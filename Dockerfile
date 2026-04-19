FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Reproducible splits + multi-model train (best artifact; MLflow skipped in image)
RUN python scripts/create_splits.py && \
    python scripts/train_experiments.py \
      --train-csv data/splits/train.csv \
      --test-csv data/splits/test.csv \
      --skip-mlflow

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]