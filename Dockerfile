FROM ubuntu:20.04

USER root

WORKDIR /app

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && pip3 install --upgrade pip \
    && apt-get install -y uvicorn

COPY api.py *.txt ./
COPY model_training_process ./model_training_process

ENV PYTHON_PATH=/app

RUN pip install --no-cache-dir --user -r requirements_nvidia.txt --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --user -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
