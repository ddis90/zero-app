FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including Playwright deps + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs /state

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PAPER_TRADE=true
ENV STATE_DIR=/state

# Health check via HTTP endpoint
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run the trading agent
CMD ["python", "-m", "src.main"]
