#!/bin/sh
set -eu

# Full images keep their preloaded model in an immutable image layer. Slim
# images use the mounted data volume so an explicit user download survives a
# container recreation.
if [ "${LAB_IMAGE_VARIANT:-full}" = "slim" ]; then
  export SENTENCE_TRANSFORMERS_HOME=/data/models
else
  export SENTENCE_TRANSFORMERS_HOME=/opt/tiny-rag-models
fi

exec "$@"
