#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
IMAGE_NAME=${1:-foundationpose-env:test}

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}

docker build \
  --network=host \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --build-arg ALL_PROXY="${ALL_PROXY:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  --build-arg http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  --build-arg https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  --build-arg all_proxy="${all_proxy:-${ALL_PROXY:-}}" \
  --build-arg no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  -t "${IMAGE_NAME}" \
  -f "${PROJECT_DIR}/Dockerfile" \
  "${PROJECT_DIR}"
