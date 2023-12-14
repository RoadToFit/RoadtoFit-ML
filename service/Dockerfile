# Use the official lightweight Node.js 16 image.
# https://hub.docker.com/_/node
FROM node:18-slim

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy application dependency manifests to the container image.
# A wildcard is used to ensure both package.json AND yarn.lock are copied.
# Copying this separately prevents re-running yarn install on every code change.
COPY package.json yarn.lock ./

# Install dependencies.
RUN yarn install

# Copy local code to the container image.
COPY . .

# Set the environment to production
ENV NODE_ENV production

# Expose port
EXPOSE 4000

# Run the web service on container startup.
CMD [ "node", "index.js" ]