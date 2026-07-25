# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Instalar Poetry
RUN pip install --no-cache-dir poetry

# Copiar solo los archivos de dependencias primero (aprovecha cache de capas)
COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --only main --no-interaction --no-ansi

# Copiar el resto del proyecto
COPY src ./src
COPY artifacts ./artifacts
COPY data ./data
COPY README.md ./

# Instalar el paquete del proyecto
RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 8000

CMD ["uvicorn", "financial_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
