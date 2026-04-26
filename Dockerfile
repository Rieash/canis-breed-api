# Build 6 - Flask dev server for debugging
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD python -c "from app_v4 import app; app.run(host='0.0.0.0', port=5000, debug=True)"
