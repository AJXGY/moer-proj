FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL maintainer="MTT Training Test"
LABEL description="5.1.6 MTT Training Run Verification"

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt /workspace/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制训练脚本
COPY preflight_check.py /workspace/
COPY train_runner.py /workspace/
COPY train_summarize.py /workspace/
COPY generate_training_charts.py /workspace/
COPY train_config.json /workspace/

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 创建输出目录
RUN mkdir -p /workspace/artifacts /workspace/logs

# 默认命令
CMD ["bash"]
