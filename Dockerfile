FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Inštalácia Python dependencies pre vLLM (najrýchlejší runtime pre LLM inference)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==0.17.0 \
    vllm==0.5.0 \
    transformers==4.41.0 \
    tokenizers==0.15.0 \
    sentencepiece==0.2.0 \
    accelerate==0.30.0 \
    runpod==0.10.4

# Kopíruj handler
COPY handler.py .

# Exposuj port (voliteľne)
EXPOSE 8000

# Spustiteľ
CMD ["python3", "handler.py"]
