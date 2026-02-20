#!/bin/bash
echo "Installing CPU-only PyTorch..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
echo "Starting Gunicorn..."
# Use $PORT provided by Cloud Run, fallback to 8080 if not set
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120
