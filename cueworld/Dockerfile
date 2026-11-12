FROM python:3.10-slim-bookworm

# ヘッドレスの PyBullet（EGL / TinyRenderer）に必要な共有ライブラリ
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ビルドコンテキストは cueworld ディレクトリ。
# import cueworld が通るよう、親ディレクトリの下にパッケージ名で置く。
WORKDIR /opt/src
COPY . /opt/src/cueworld
ENV PYTHONPATH=/opt/src
WORKDIR /opt/src

CMD ["sleep", "infinity"]
