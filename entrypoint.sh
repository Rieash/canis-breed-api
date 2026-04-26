#!/bin/bash
set -e

# Get port from environment or default to 5000
if [ -z "$PORT" ]; then
  PORT=5000
fi

echo "Starting server on port $PORT"
exec gunicorn app_v4:app --workers 1 --threads 4 --bind 0.0.0.0:$PORT
