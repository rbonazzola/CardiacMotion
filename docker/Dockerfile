# FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y make
RUN apt-get install -y --reinstall build-essential
RUN apt-get install -y libboost-dev

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

# install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.9 environment.
RUN /home/user/miniconda/bin/conda install conda-build
RUN /home/user/miniconda/bin/conda create -y --name cardiac_motion python=3.9
RUN /home/user/miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=cardiac_motion
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

ENV TORCH=1.10
ENV CUDA=cu102
RUN conda install pytorch=${TORCH} torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install tqdm -c conda-forge
RUN conda install jupyter -c anaconda
RUN conda install pytorch-lightning -c conda-forge
RUN conda install mlflow -c conda-forge

RUN conda install pyg -c pyg -c conda-forge

RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

#RUN apt-get update 
RUN sudo apt-get install && sudo apt-get install -y module-init-tools

# RUN sudo apt-get install linux-headers-4.4.0-1122-aws

# RUN wget https://us.download.nvidia.com/tesla/440.118.02/NVIDIA-Linux-x86_64-440.118.02.run
# RUN sudo bash NVIDIA-Linux-x86_64-440.118.02.run -s

# Install vtk and others
RUN pip install vtk meshio nvgpu

# Install psbody-mesh
WORKDIR /home/user
RUN git clone https://github.com/MPI-IS/mesh.git
WORKDIR mesh
RUN make all
