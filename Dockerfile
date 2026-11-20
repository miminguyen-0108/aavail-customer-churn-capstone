FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /service
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY aavail/ ./aavail/
COPY scripts/ ./scripts/
COPY data/ ./data/
RUN python scripts/run_pipeline.py
RUN useradd --create-home --uid 10001 appuser && mkdir -p /service/logs && chown -R appuser:appuser /service/logs
USER appuser
EXPOSE 8000
CMD ["uvicorn", "aavail.api:app", "--host", "0.0.0.0", "--port", "8000"]
