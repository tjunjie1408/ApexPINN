FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/ml/code
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.1
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra cuda --no-editable --python /usr/local/bin/python

ENTRYPOINT ["/opt/venv/bin/python", "-m", "apexpinn.cloud.sagemaker_worker"]
