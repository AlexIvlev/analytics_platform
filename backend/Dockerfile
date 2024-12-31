FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential=12.9 \
    software-properties-common=0.99.30-4.1~deb12u1\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

COPY . /backend

ENV PYTHONPATH=/backend

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
