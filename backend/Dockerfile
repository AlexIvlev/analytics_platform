FROM python:3.11-alpine

RUN apk add --no-cache \
    build-base \
    g++ \
    gcc \
    libffi-dev \
    musl-dev \
    python3-dev \
    py3-pip

WORKDIR /backend

COPY . /backend

ENV PYTHONPATH=/backend

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
