# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install project dependencies
RUN pip install --no-cache-dir ".[all]"

# Create non-root user
RUN useradd -m -u 1000 repofellow
USER repofellow

# Set Python path
ENV PYTHONPATH=/app/src

# Command to run the application
CMD ["python", "-m", "repofellow"]
