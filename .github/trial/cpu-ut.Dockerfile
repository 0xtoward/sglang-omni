# Trial only: a CPU-only image that holds the unit-test environment.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_TORCH_BACKEND=cpu \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates curl git ffmpeg sox time build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . /src
WORKDIR /src

RUN uv venv /opt/venv -p 3.12 \
    && uv pip install --no-cache --python /opt/venv/bin/python -e '.[minicpm-o,fun-cosyvoice3]' \
    && uv pip install --no-cache --python /opt/venv/bin/python \
        descript-audio-codec==1.0.0 descript-audiotools==0.7.2 \
    && uv pip install --no-cache --python /opt/venv/bin/python --no-deps qwen-tts==0.1.1 \
    && git clone --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git /opt/venv/src/CosyVoice \
    && git -C /opt/venv/src/CosyVoice checkout --detach 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
    && git -C /opt/venv/src/CosyVoice submodule update --init --depth=1 third_party/Matcha-TTS \
    && printf '%s\n' /opt/venv/src/CosyVoice /opt/venv/src/CosyVoice/third_party/Matcha-TTS \
        > /opt/venv/lib/python3.12/site-packages/cosyvoice.pth
