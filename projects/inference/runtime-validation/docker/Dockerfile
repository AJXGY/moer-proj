FROM python:3.10-slim

WORKDIR /workspace/5.1.5

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/5.1.5

ENTRYPOINT ["bash", "/workspace/5.1.5/run_515_suite.sh"]
