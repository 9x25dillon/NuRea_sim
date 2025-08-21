FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl is handy for quick debugging inside the container
RUN apt-get update && apt-get install -y --no-install-recommends curl \
     && rm -rf /var/lib/apt-lists/*

WORKDIR /workspace

# Install runtime deps from minimal requirements
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Fall back to editable install if you ship a pyproject (safe if present)
COPY . /workspace
RUN if [ -f "pyproject.toml" ]; then pip install --no-cache-dir -e . || true; fi

CMD ["python", "-u", "matrix_orchestrator.py", "--plan", "/workspace/plan.json"]
