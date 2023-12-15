# Use the official lightweight Node.js 16 image.
# https://hub.docker.com/_/node
FROM python:3.10

# Create and change to the app directory.
WORKDIR /code

# Copy application dependency manifests to the container image.
COPY ./requirements.txt /code/requirements.txt

# Install dependencies.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy local code to the container image.
COPY . /code/

# Run the web service on container startup.
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000" ]
