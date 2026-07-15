FROM node:22-slim AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM node:22-slim AS guides-build
WORKDIR /guides
COPY learning_materials/package.json learning_materials/package-lock.json ./
RUN npm ci
COPY learning_materials/ ./
RUN npm run build

FROM python:3.12-slim AS lab
ARG LAB_IMAGE_VARIANT=full
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TINY_RAG_LAB_DATA_DIR=/data \
    LAB_IMAGE_VARIANT=${LAB_IMAGE_VARIANT}
WORKDIR /app
COPY pyproject.toml README.md ./
# The local lab has no GPU execution path.  Install the official CPU-only
# wheel first so sentence-transformers cannot resolve a CUDA/NVIDIA runtime
# transitively on Linux. Keeping it before application source preserves this
# expensive, CPU-only layer while the lab code is refined.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
COPY tiny_rag_lab ./tiny_rag_lab
COPY scripts ./scripts
COPY assets/seed/v1 /opt/tiny-rag-lab/seeds/v1
COPY docker-entrypoint.sh /usr/local/bin/tiny-rag-lab-entrypoint
RUN chmod +x /usr/local/bin/tiny-rag-lab-entrypoint \
    && pip install --no-cache-dir '.[qdrant]'
COPY --from=web-build /web/dist /app/web-dist
COPY --from=guides-build /guides/.vitepress/dist /app/web-dist/docs
# Full prepares the existing default embedder at build time; slim defers it.
RUN if [ "$LAB_IMAGE_VARIANT" = "full" ]; then SENTENCE_TRANSFORMERS_HOME=/opt/tiny-rag-models python -c "from tiny_rag_lab.embeddings import SentenceTransformerEmbedder; SentenceTransformerEmbedder()"; fi
EXPOSE 8000
ENTRYPOINT ["tiny-rag-lab-entrypoint"]
CMD ["uvicorn", "tiny_rag_lab.web_api:create_packaged_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
