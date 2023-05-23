FROM ghcr.io/allenai/cuda:11.5-cudnn8-dev-ubuntu20.04-v1.2.13 AS pytorch_installer

ENV APP_HOME /root/
WORKDIR $APP_HOME

## Install some libraries required by opencv, also nano
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 nano  -y

RUN conda create -n rearrange -y python=3.9

ENV PYTHON=/opt/miniconda3/envs/rearrange/bin/python
ENV PIP=/opt/miniconda3/envs/rearrange/bin/pip

RUN $PIP install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html;

RUN /opt/miniconda3/bin/conda install -n rearrange -y -c conda-forge jsonnet ipython
RUN /opt/miniconda3/bin/conda clean -ya

COPY requirements.txt /root/requirements.txt

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/rearrange/pipsrc; \
    $PIP install -r /root/requirements.txt \
    && $PIP install -e "git+https://github.com/openai/CLIP.git@e184f608c5d5e58165682f7c332c3a8b4c1545f2#egg=clip" \
)

COPY . /root/ai2thor-rearrangement/

WORKDIR /root/ai2thor-rearrangement/

ENTRYPOINT ["bash", "-l"]