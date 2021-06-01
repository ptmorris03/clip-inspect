FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ENV LANG C.UTF-8

RUN apt-get update && apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-pandas \
    wget

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

WORKDIR /opt/colab

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    jupyterlab jupyter_http_over_ws ipywidgets google-colab\
    && jupyter serverextension enable --py jupyter_http_over_ws \
    && jupyter nbextension enable --py widgetsnbextension

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    numpy \
    scipy \
    matplotlib \
    ipython \
    jupyter \
    sympy \
    nose \
    tqdm \
    scikit-learn \ 
    h5py \
    ftfy \
    regex

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    transformers

#moviepy
RUN apt-get -y update && apt-get -y install ffmpeg imagemagick

RUN apt-get -y install fonts-liberation

RUN apt-get install -y git locales && \
    locale-gen en_US.UTF-8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8

RUN git clone https://github.com/Zulko/moviepy.git /var/src/moviepy
RUN cd /var/src/moviepy/ && pip3 install .[optional]

# modify ImageMagick policy file so that Textclips work correctly.
RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml 

ARG COLAB_PORT=8081
EXPOSE ${COLAB_PORT}
ENV COLAB_PORT ${COLAB_PORT}

CMD jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --allow-root --port $COLAB_PORT --NotebookApp.port_retries=0 --ip 0.0.0.0
