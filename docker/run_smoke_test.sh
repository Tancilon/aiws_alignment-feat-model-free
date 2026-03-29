#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT_DIR=$(cd -- "${PROJECT_DIR}/.." && pwd)
IMAGE_NAME=${1:-foundationpose-env:test}
SCENE_DIR=${FOUNDATIONPOSE_SCENE_DIR:-${ROOT_DIR}/data/tmp_output/foundationpose}
MESH_FILE=${FOUNDATIONPOSE_MESH_FILE:-${SCENE_DIR}/mesh/gaiban.obj}
DEBUG_DIR_HOST=${FOUNDATIONPOSE_DEBUG_DIR_HOST:-${ROOT_DIR}/data/tmp_output/foundationpose_debug_docker}
WARP_CACHE_HOST=${FOUNDATIONPOSE_WARP_CACHE_HOST:-/tmp/foundationpose-warp-cache}

mkdir -p "${DEBUG_DIR_HOST}" "${WARP_CACHE_HOST}"

docker run --rm --gpus all \
  -e FOUNDATIONPOSE_SCENE_DIR=/workspace/scene \
  -e FOUNDATIONPOSE_MESH_FILE=/workspace/scene/mesh/$(basename "${MESH_FILE}") \
  -e FOUNDATIONPOSE_DEBUG_DIR=/workspace/debug \
  -e WARP_CACHE_DIR=/tmp/warp-cache \
  -v "${SCENE_DIR}:/workspace/scene:ro" \
  -v "${DEBUG_DIR_HOST}:/workspace/debug" \
  -v "${WARP_CACHE_HOST}:/tmp/warp-cache" \
  "${IMAGE_NAME}" \
  python /opt/foundationpose/docker/smoke_test.py
