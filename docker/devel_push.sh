#!/bin/bash

docker container rm clip-inspect

docker run \
    -it --init \
    --name "clip-inspect" \
    pmorris2012/clip-inspect:jax-base python3 -c "print('pushing code and installing clip-inspect python3 library')"

docker cp . clip-inspect:/clip-inspect

docker commit clip-inspect pmorris2012/clip-inspect:jax-base

docker container stop clip-inspect

docker container rm clip-inspect
