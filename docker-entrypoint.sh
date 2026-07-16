#!/bin/sh
set -eu

# Full images keep their preloaded model in an immutable image layer. Slim
# images use the mounted data volume so an explicit user download survives a
# container recreation.
if [ "${LAB_IMAGE_VARIANT:-full}" = "slim" ]; then
  export HF_HOME=/data/models
  export HF_HUB_CACHE=/data/models/hub
else
  export HF_HOME=/opt/tiny-rag-models
  export HF_HUB_CACHE=/opt/tiny-rag-models/hub
fi

exec "$@"
