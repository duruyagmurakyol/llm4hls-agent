FROM xilinx/xilinx_runtime_base:alveo-2023.2-ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Match the organiser-provided Vitis container userspace. Vitis itself is not
# baked into the image; the host's Xilinx 2025.2 installation is bind-mounted
# read-only at runtime by docker/run-vitis.sh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git sudo locales \
    && locale-gen en_US.UTF-8 \
    && dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gparted xinetd gawk gcc g++ build-essential make cmake automake \
        openssl libssl-dev flex bison autoconf libtool texinfo zlib1g-dev \
        iproute2 net-tools diffstat chrpath socat tar unzip gzip tofrodos \
        lsb-release libftdi1 libftdi1-2 openssh-client debianutils \
        iputils-ping libegl1-mesa libsdl1.2-dev cpio gnupg perl xvfb \
        gcc-multilib python3 python3-pip python3-git python3-jinja2 \
        python3-pexpect xz-utils liberror-perl xtrans-dev \
        libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev \
        libxcb-shape0-dev libxcb-xkb-dev util-linux sysvinit-utils \
        ocl-icd-libopencl1 opencl-headers ocl-icd-opencl-dev \
        libncurses-dev libncurses5 libncurses5-dev libncursesw5 \
        libncursesw5-dev libncurses5:i386 libtinfo5 lib32stdc++6 \
        libstdc++6:i386 libgtk2.0-0:i386 libfontconfig1:i386 \
        libx11-6:i386 libxext6:i386 libxrender1:i386 libsm6:i386 \
        zlib1g:i386 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN echo "dash dash/sh boolean false" | debconf-set-selections \
    && dpkg-reconfigure -f noninteractive dash \
    && echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace/llm4hls-agent

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker/entrypoint.sh docker/run-vitis.sh \
    && python3 -m compileall -q agent scripts

ENTRYPOINT ["/workspace/llm4hls-agent/docker/entrypoint.sh"]
