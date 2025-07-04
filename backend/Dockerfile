# Multi-stage build for Python FastAPI backend
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        gcc \
        git \
        docker.io \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage /app/logs \
    && chown -R app:app /app

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    debugpy \
    ipython \
    jupyter

# Copy development configuration
COPY --chown=app:app ./config/development.py /app/config/development.py

# Expose port
EXPOSE 8000

# Set user
USER app

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    gunicorn

# Copy production configuration
COPY --chown=app:app ./config/production.py /app/config/production.py
COPY --chown=app:app ./config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Remove development files
RUN find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} + \
    && rm -rf .pytest_cache .coverage

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set user
USER app

# Production command with Gunicorn
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]

# Testing stage
FROM development as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-mock \
    factory-boy \
    freezegun

# Copy test configuration
COPY --chown=app:app ./tests /app/tests
COPY --chown=app:app ./pytest.ini /app/pytest.ini

# Run tests
CMD ["pytest", "-v", "--cov=app", "--cov-report=html", "--cov-report=term"]

# Security scanning stage
FROM base as security

# Install security tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep

# Run security scans
CMD ["sh", "-c", "bandit -r app/ && safety check && semgrep --config=auto app/"]