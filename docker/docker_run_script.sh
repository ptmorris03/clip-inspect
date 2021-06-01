#!/bin/bash

docker run \
    -it --rm --init \
    --shm-size=16G \
    --gpus all \
    -v /media/mpcrpaul/fastdata/colab:/data \
    -v $(pwd):/code \
    colab-local-runtime python3 /code/"$@"
