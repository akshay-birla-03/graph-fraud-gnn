FROM python:3.11-slim

WORKDIR /app

# CPU-only torch to keep the image small and offline-friendly.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir torch>=2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[dev]"

COPY tests ./tests

CMD ["graphfraud"]
