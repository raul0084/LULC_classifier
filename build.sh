#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="sentinel-2-beach-classifier"

echo "Building Docker image: ${IMAGE_NAME}"

docker build \
  -t "${IMAGE_NAME}" \
  "${PROJECT_ROOT}"

echo "Build complete."