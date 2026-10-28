FROM python:3.11-slim

WORKDIR /app

COPY api/ ./api/
COPY models/ ./models/

RUN pip install --no-cache-dir fastapi uvicorn pandas joblib scikit-learn numpy

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]