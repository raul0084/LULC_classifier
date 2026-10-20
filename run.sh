#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="sentinel-2-beach-classifier"
CONTAINER_NAME="sentinel-2-beach-classifier"

# Build the image if it doesn't exist yet
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image not found. Building it first..."
  "$(dirname "$0")/build.sh"
fi

echo "Starting Streamlit app at http://localhost:8501"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -p 8501:8501 \
  "${IMAGE_NAME}"