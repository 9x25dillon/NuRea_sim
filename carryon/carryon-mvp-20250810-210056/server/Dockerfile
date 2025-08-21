FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -U pip &&         pip install --no-cache-dir -e .
COPY app ./app
ENV CARRYON_DB_URL=sqlite:///data/carryon.db
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
