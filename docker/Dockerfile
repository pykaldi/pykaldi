# Install pykaldi in a Docker container
# With code from
# https://github.com/RaRe-Technologies/gensim/blob/develop/docker/Dockerfile
FROM ubuntu:16.04

WORKDIR /root

# Update ubuntu and install essentials
RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    python2.7 \
    autoconf \
    automake \
    cmake \
    curl \
    g++ \
    git \
    libatlas3-base \
    libtool \
    make \
    pkg-config \
    subversion \
    unzip \
    wget \
    zlib1g-dev

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Install necessary python packages
RUN pip install --upgrade pip \
    numpy \
    setuptools \
    pyparsing \
    jupyter \
    ninja

# Download pykaldi from git and install
RUN git clone "https://github.com/pykaldi/pykaldi.git" pykaldi \
    && cd pykaldi \
    && cd tools \
    && ./check_dependencies.sh \
    && ./install_protobuf.sh \
    && ./install_clif.sh \
    && ./install_kaldi.sh \
    && cd .. \
    && python setup.py install

# # Add run permission to startup script
RUN chmod +x pykaldi/docker/start_jupyter_notebook.sh
RUN chmod +x pykaldi/docker/update_kaldi.sh
RUN chmod +x pykaldi/docker/update_pykaldi.sh

# # Define the starting command for this container and expose its running port
CMD sh -c '/root/pykaldi/docker/start_jupyter_notebook.sh 9000'
EXPOSE 9000
