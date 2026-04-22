#!/bin/sh
exec gunicorn app_v3:app --workers 1 --threads 4 --bind 0.0.0.0:${PORT:-5000}
