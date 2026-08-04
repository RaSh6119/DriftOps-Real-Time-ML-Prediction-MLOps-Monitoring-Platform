FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./backend/app
COPY ml ./ml
COPY models ./models
COPY data ./data
COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns

WORKDIR /app/backend/app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]