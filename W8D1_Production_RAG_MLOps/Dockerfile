
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/W7D5_Multi_Document_RAG
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

WORKDIR /app

COPY W8D1_Production_RAG_MLOps/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY W8D1_Production_RAG_MLOps/app /app/W8D1_Production_RAG_MLOps/app
COPY W7D5_Multi_Document_RAG /app/W7D5_Multi_Document_RAG

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/W8D1_Production_RAG_MLOps", "--host", "0.0.0.0", "--port", "8000"]
