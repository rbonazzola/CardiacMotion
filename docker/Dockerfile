# FROM ubuntu:18.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
# nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    zip \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y tmux
RUN apt-get install -y make
RUN apt-get install -y --reinstall build-essential
RUN apt-get install -y libboost-dev

RUN apt-get update -y && \
    apt-get install -y build-essential libfuse-dev libcurl4-openssl-dev libxml2-dev pkg-config libssl-dev mime-support automake libtool wget tar git unzip
RUN apt-get install lsb-release -y  && apt-get install zip -y && apt-get install vim -y

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
ENV CUDA=cu113
RUN conda install pytorch=${TORCH} torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install tqdm -c conda-forge
RUN conda install jupyter -c anaconda
RUN conda install pytorch-lightning -c conda-forge
RUN conda install mlflow==1.23 -c conda-forge

RUN conda install pyg -c pyg -c conda-forge

RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


# I am including this in case it's needed, but at the moment I'm using the host's Nvidia drivers.
# RUN apt-get update 
# RUN sudo apt-get install && sudo apt-get install -y module-init-tools
# RUN sudo apt-get install linux-headers-4.4.0-1122-aws
# RUN wget https://us.download.nvidia.com/tesla/440.118.02/NVIDIA-Linux-x86_64-440.118.02.run
# RUN sudo bash NVIDIA-Linux-x86_64-440.118.02.run -s

# Install vtk and others
RUN pip install vtk meshio vedo trimesh pyvista

RUN pip install nvgpu

# Install psbody-mesh
WORKDIR /home/user
RUN git clone https://github.com/MPI-IS/mesh.git
WORKDIR mesh
RUN make all

# Set up read-only SSH keys for personal GitHub repositories
RUN mkdir /home/user/.ssh
COPY --chown=user ssh_keys/* /home/user/.ssh/
WORKDIR /app
RUN touch /home/user/.ssh/known_hosts
RUN ssh-keyscan github.com >> /home/user/.ssh/known_hosts

# Clone personal repositories
RUN ssh-agent bash -c 'ssh-add /home/user/.ssh/cardiac_motion; git clone git@github.com:rbonazzola/CardiacMotionRL.git'
WORKDIR /app/CardiacMotionRL
RUN git checkout development
RUN ssh-agent bash -c 'ssh-add /home/user/.ssh/vtk_helpers; git submodule update --init --recursive'
WORKDIR /app/CardiacMotionRL/utils/VTKHelpers
RUN git checkout development
WORKDIR /home/user

# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip
# RUN ./aws/install -i /home/user/aws-cli -b /home/user/bin

RUN pip install s3cmd
COPY --chown=user aws/.s3cfg.gpg /home/user

RUN conda install -c conda-forge awscli

USER root
## Install S3 Fuse
RUN rm -rf /usr/src/s3fs-fuse
RUN git clone https://github.com/s3fs-fuse/s3fs-fuse/ /usr/src/s3fs-fuse
WORKDIR /usr/src/s3fs-fuse 
RUN ./autogen.sh && ./configure && make && make install
RUN apt-get install fuse

## Create folder
USER user
RUN mkdir $HOME/s3

## Set the directory where you want to mount your s3 bucket
ENV S3_MOUNT_DIRECTORY=$HOME/s3

## Replace with your s3 bucket name
ENV S3_BUCKET_NAME=users-rbonazzola

## Entry Point
ADD --chown=user start-script.sh $HOME/start-script.sh
RUN chmod 755 $HOME/start-script.sh 
COPY --chown=user aws/.passwd-s3fs.gpg $HOME
WORKDIR $HOME
CMD ["./start-script.sh"]
