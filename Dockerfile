FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install all Python dependencies (torch uses CPU-only via --extra-index-url in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run sets PORT env var
ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
