# Multi-step Docker build

# Build stage
FROM python:3.10-slim AS builder

# Create and change to the app directory.
WORKDIR /code

# Copy application dependency manifests to the container image.
COPY ./requirements.txt /code/requirements.txt

# Install dependencies.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install uvicorn

# Runtime stage
FROM python:3.10-slim

WORKDIR /code

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy local code to the container image.
COPY . .

ENV PYTHONUNBUFFERED true

# Run the web service on container startup.
CMD [ "python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000" ]
