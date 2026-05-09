FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies required for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed models results

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=15s --timeout=10s --start-period=360s --retries=10 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint: run preprocessing, training, evaluation, then launch Streamlit
CMD ["bash", "-c", \
    "python scripts/preprocess_data.py && \
     python scripts/train.py && \
     python scripts/evaluate.py && \
     streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
