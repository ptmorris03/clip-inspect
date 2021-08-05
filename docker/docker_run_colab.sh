#!/bin/bash

docker run \
    -it --rm --init \
    --shm-size=16G \
    --gpus all \
    -p 8081:8081 \
    -v $(pwd):/opt/colab \
    colab-local-runtime

