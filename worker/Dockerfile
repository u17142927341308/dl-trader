FROM python:3.11-slim
WORKDIR /app
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/tasks_runner.py .
COPY worker/tasks.py .
COPY backend/app /app/app
ENV PYTHONPATH=/app:/app/app
CMD ["celery", "-A", "tasks_runner.celery_app", "worker", "-l", "INFO", "--concurrency=2"]
