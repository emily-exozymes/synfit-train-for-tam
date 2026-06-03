FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    "setuptools<81" \
    pip \
    git \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH

# Pin HF cache to a known location so build-time pre-cache and runtime
# load look at the same place (Tamarind runtime can run as a non-root user
# with HOME=/tmp, which would hide a default ~/.cache/huggingface/ cache).
ENV HF_HOME=/opt/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache

# PyTorch 2.4.0 + CUDA 12.1 (matches SynFit README exactly).
# Use --extra-index-url so PyPI stays in the search path for torch's
# transitive deps (fsspec, sympy, etc.) - --index-url alone replaces PyPI
# entirely and the build fails on missing fsspec.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0

# Full SynFit training stack
RUN pip install --no-cache-dir \
    triton==3.0.0 \
    "transformers>=4.40,<4.50" \
    "tokenizers>=0.19,<0.21" \
    "huggingface_hub>=0.23" \
    "accelerate>=0.30" \
    "peft>=0.10" \
    "safetensors>=0.4" \
    "datasets>=2.18" \
    numpy==1.26.4 \
    scipy==1.15.3 \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "einops>=0.7" \
    matplotlib==3.10.6 \
    seaborn==0.13.2 \
    biopython==1.85 \
    PyYAML \
    "tensorboard>=2.16" \
    tqdm

# Pre-cache ESM2-650M (no internet at runtime). Files land in
# /opt/hf_cache/hub/ because we set HF_HOME above. The runtime envVars in
# config.json point HF_HOME at the same path and set HF_HUB_OFFLINE=1 so
# transformers loads from the cache without any network call.
RUN python -c "from transformers import EsmForMaskedLM, EsmTokenizer; \
    EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')"

# Make the cache world-readable in case Tamarind runs as a non-root user.
RUN chmod -R a+rX /opt/hf_cache

RUN mkdir -p inputs out && chmod -R 777 /app
