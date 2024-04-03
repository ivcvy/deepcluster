FROM ubuntu:22.04
# FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    vim \
    wget \
    curl \
    cmake \
    build-essential

RUN curl -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh &&\
    chmod +x miniconda.sh &&\
    ./miniconda.sh -b -p /opt/conda &&\
    rm miniconda.sh

RUN export PATH=$PATH:/opt/conda/bin &&\
    conda create -n pytorch3 python=3

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pytorch3", "/bin/bash", "-c"]
ENV PATH="/opt/conda/bin:${PATH}"

RUN export CMAKE_PREFIX_PATH=/opt/conda/ &&\
    conda install numpy mkl setuptools cmake cffi scikit-learn &&\
    apt-get -y install gcc libblas-dev liblapack-dev &&\
    conda install -c soumith magma-cuda80 &&\
    conda install pytorch==1.3.1 torchvision==0.4.2 cpuonly -c pytorch

RUN conda install faiss-cpu==1.3.0 -c pytorch &&\
    conda init bash

RUN pip install bpython future typer pandas
RUN echo "source activate pytorch3" >> /root/.bashrc

ENV PYTHONPATH="/usr/src/app/faiss-1.3.0/python/:${PYTHONPATH}"
COPY . .
