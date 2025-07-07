FROM asia-docker.pkg.dev/colab-images/public/runtime:release-colab_20250603-060055_RC00

# 必要な環境変数を設定して非対話的にインストール
ENV DEBIAN_FRONTEND=noninteractive

# パッケージのアップデートとインストール
RUN apt -y update && \
    apt -y install libopencv-dev tzdata git && \
    apt -y install iputils-ping net-tools && \
    apt -y install build-essential -y && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Pythonライブラリをインストール(RTX5000シリーズサポート)
    pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -U transformers accelerate bitsandbytes



# Jupyter を起動するエントリポイントを指定
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--port=8080", \
"--NotebookApp.allow_origin=https://colab.research.google.com", \
"--NotebookApp.token='123123123'" ]
