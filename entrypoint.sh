#!/bin/bash
exec gunicorn app_v4:app --workers 1 --threads 4 --bind 0.0.0.0:5000
