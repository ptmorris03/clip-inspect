#!/bin/bash

docker run \
    -it --rm --init \
    --shm-size=16G \
    --gpus all \
    -v $(pwd):/code \
    --name "clip-inspect"
    pmorris2012/clip-inspect:jax-base python3 /code/"$@"
