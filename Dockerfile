# syntax=docker/dockerfile:1.3

FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Use cache for pip downloads across builds
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Preinstall torch to avoid stanza hash mismatch
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir torch==2.1.2

# Install remaining requirements with retry and long timeout
RUN --mount=type=cache,target=/root/.cache \
    pip install --default-timeout=1000 --retries=10 --no-cache-dir -r /code/requirements.txt

# Copy the actual app code after dependencies to avoid breaking the cache
COPY ./app /code/app

# Expose FastAPI port
EXPOSE 8000

# Start with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
