FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ENV LANG C.UTF-8

RUN apt update && apt -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --fix-missing\
    python3 \
    python3-pip \
    wget \
    git \
    ffmpeg

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

COPY . /clip-inspect
#RUN git clone -b jax git@https://github.com/pmorris2012/clip-inspect.git
WORKDIR /clip-inspect

RUN pip3 install -r requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN pip3 install -e .
