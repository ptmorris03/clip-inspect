#!/bin/bash

docker run \
    -it --rm --init \
    --shm-size=16G \
    --gpus all \
    -p 8081:8081 \
    -v /media/mpcrpaul/fastdata/colab:/opt/colab \
    -v $(pwd):/code \
    colab-local-runtime

