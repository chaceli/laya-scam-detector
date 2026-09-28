FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
      "onnxruntime>=1.18" \
      "tokenizers>=0.15" \
      "numpy>=1.24" \
      "huggingface_hub>=0.24" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27"

# Fine-tuned multilingual checkpoint, shipped as a GitHub Release asset.
# Override MODEL_URL to pin a different release or a Hugging Face URL.
ARG MODEL_URL=https://github.com/chaceli/laya-scam-detector/releases/download/v1.0.0/laya-onnx-multilingual-finetuned-fp16.tar.gz
RUN mkdir -p /app/models \
 && curl -fL --retry 3 -o /tmp/model.tgz "$MODEL_URL" \
 && tar xzf /tmp/model.tgz -C /app/models \
 && rm /tmp/model.tgz \
 && test -f /app/models/laya-onnx-multilingual-finetuned-fp16/model.onnx

COPY server/ ./server/
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY web/ ./web/
COPY main.py ./

# Single-checkpoint deployment: English disabled, fine-tuned model only.
ENV LAYA_ENGLISH_DIR="" \
    LAYA_MULTILINGUAL_DIR=/app/models/laya-onnx-multilingual-finetuned-fp16 \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=7860

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]