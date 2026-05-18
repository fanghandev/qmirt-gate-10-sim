FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1

RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://\(archive\|security\).ubuntu.com/ubuntu|mirror://mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libice6 \
        libsm6 \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --upgrade --ignore-installed pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
        opengate[novis] \
        openpyxl \
        polars \
        uproot \
        plotly \
        nibabel \
        SimpleITK \
        scipy \
        numpy \
        matplotlib \
        tqdm

RUN python3 - <<'PY'
import opengate_core  # noqa: F401
PY

WORKDIR /workspace

CMD ["python3"]