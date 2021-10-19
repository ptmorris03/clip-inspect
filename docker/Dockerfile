FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

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

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.72

# install python3-pip
RUN apt update && apt install python3-pip -y

# install dependencies via pip
RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    pip3 install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


COPY . /clip-inspect
#RUN git clone -b jax git@https://github.com/pmorris2012/clip-inspect.git
WORKDIR /clip-inspect

RUN pip3 install -r requirements.txt

RUN pip3 install -e .
