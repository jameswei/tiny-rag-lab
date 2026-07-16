from pathlib import Path


def test_studio_image_is_cpu_only_and_uses_seed_v2():
    dockerfile = Path("Dockerfile").read_text()

    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.7.1+cpu" in dockerfile
    assert "COPY assets/seed/v2 /opt/tiny-rag-lab/seeds/v2" in dockerfile
    assert "nvidia-" not in dockerfile.lower()


def test_full_bundles_both_pinned_models_while_slim_defers_them():
    dockerfile = Path("Dockerfile").read_text()

    assert 'if [ "$LAB_IMAGE_VARIANT" = "full" ]' in dockerfile
    assert "SentenceTransformerEmbedder()" in dockerfile
    assert "CrossEncoderReranker.ensure_default_model(local_files_only=False)" in dockerfile
    assert dockerfile.count('if [ "$LAB_IMAGE_VARIANT" = "full" ]') == 1


def test_slim_model_downloads_use_the_persistent_data_volume():
    entrypoint = Path("docker-entrypoint.sh").read_text()

    assert "export HF_HOME=/data/models" in entrypoint
    assert "export HF_HUB_CACHE=/data/models/hub" in entrypoint
    assert "SENTENCE_TRANSFORMERS_HOME" not in entrypoint


def test_model_layer_precedes_frequently_changed_runtime_assets():
    dockerfile = Path("Dockerfile").read_text()

    model_layer = dockerfile.index("CrossEncoderReranker.ensure_default_model")
    assert model_layer < dockerfile.index("COPY assets/seed/v2")
    assert model_layer < dockerfile.index("COPY --from=web-build")
    assert model_layer < dockerfile.index("COPY --from=guides-build")
    assert model_layer < dockerfile.index("COPY docker-entrypoint.sh")


def test_python_ci_installs_cpu_only_torch_without_syncing_cuda_lock_entries():
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert "torch==2.7.1+cpu" in workflow
    assert "uv run --no-sync pytest" in workflow
