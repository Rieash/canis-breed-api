# Build 7 - Gunicorn with proper error handling
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - --log-level debug app_v4:app
