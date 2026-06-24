FROM python:3.11-slim

WORKDIR /app

COPY ml_system/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ml_system/ .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]