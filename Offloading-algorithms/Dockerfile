# Base image: slim Python for smaller footprint
FROM python:3.12-slim

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# System deps for psycopg2 and matplotlib (fonts) + curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
        curl \
        fonts-dejavu-core \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src
COPY configs /app/configs
COPY main.py /app/main.py
COPY visualize_iov.py /app/visualize_iov.py

# Runtime configuration
ENV ENV_TYPE=db \
    PYTHONUNBUFFERED=1 \
    # Optional: override config filename at runtime (db_config.json or dummy_config.json)
    CONFIG_NAME=db_config.json

# Default command runs the DDQN agent against the selected environment
CMD ["bash", "-lc", "python /app/main.py --env ${ENV_TYPE} && sleep infinity"]

# Basic healthcheck: ensure python process is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD pgrep -f "python /app/main.py" >/dev/null || exit 1
