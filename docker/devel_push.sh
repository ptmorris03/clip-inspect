#!/bin/bash

echo "------------------------------"
echo "pushing library update"

docker container rm clip-inspect

docker run \
    -it --init \
    --name "clip-inspect" \
    pmorris2012/clip-inspect:jax-base python3 -c "print('created container')"

docker cp . clip-inspect:/clip-inspect

docker commit clip-inspect pmorris2012/clip-inspect:jax-base

docker container stop clip-inspect

docker container rm clip-inspect

echo "done pushing library update :)"
echo "------------------------------"
