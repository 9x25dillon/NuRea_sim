FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestrator application
COPY orchestrator/app /app/app

# Create artifacts directory
RUN mkdir -p /app/artifacts

# plan.json is mounted at runtime via volume
CMD ["python", "-m", "app.cli", "--plan", "/app/plan.json"]
