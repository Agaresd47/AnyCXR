FROM pytorch/pytorch:2.5.1-cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/AnyCXR

COPY . /workspace/AnyCXR

RUN pip install --upgrade pip && pip install -e .

ENTRYPOINT ["anychest-infer"]
CMD ["--help"]
