FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      git \
      locales \
      python3 \
      python3-pip \
      python3-venv \
      libfontconfig1 \
      libfreetype6 \
      libglib2.0-0t64 \
      libice6 \
      libsm6 \
      libx11-6 \
      libxau6 \
      libxcb1 \
      libxdmcp6 \
      libxext6 \
      libxft2 \
      libxi6 \
      libxrender1 \
      libxtst6 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pytest requests

WORKDIR /workspace/llm4hls-agent

COPY . .

RUN python3 -m pytest -q

ENTRYPOINT ["bash", "scripts/container_entrypoint.sh"]
CMD ["python3", "scripts/run_agent.py", "--help"]
