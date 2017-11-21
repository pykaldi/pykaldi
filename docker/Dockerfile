# Install pykaldi in a Docker container
# With code from
# https://github.com/RaRe-Technologies/gensim/blob/develop/docker/Dockerfile
FROM ubuntu:16.04

WORKDIR /root

ENV PROTOBUF_GIT "https://github.com/google/protobuf.git"
ENV NINJA_GIT "https://github.com/ninja-build/ninja.git"
ENV CLIF_GIT "-b pykaldi https://github.com/dogancan/clif/"
ENV KALDI_GIT "https://github.com/usc-sail/kaldi-pykaldi.git"
ENV PYKALDI "https://github.com/usc-sail/pykaldi.git"

# update ubuntu and install essentials
RUN apt-get update \
    && apt-get install -y \
    git=1:2.7.4-0ubuntu1 \
    cmake=3.5.1-1ubuntu3 \
    autoconf=2.69-9 \
    automake=1:1.15-4ubuntu1 \
    libtool=2.4.6-0.1 \
    curl \
    make=4.1-6 \
    g++=4:5.3.1-1ubuntu1 \
    unzip=6.0-20ubuntu1 \
    # python3=3.5.1-3 \
    # python3-pip=8.1.1-2ubuntu0.4 \
    # python3-dev=3.5.1-3 \
    python=2.7.11-1 \
    python-pip=8.1.1-2ubuntu0.4 \
    python-dev=2.7.11-1 \
    build-essential=12.1ubuntu2 \
    virtualenv=15.0.1+ds-3ubuntu1 \
    libatlas3-base=3.10.2-9 \
    wget \
    zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1 \
    subversion=1.9.3-2ubuntu1.1 \
    pkg-config=0.29.1-0ubuntu1

# Upgrade pip
RUN pip install --upgrade pip

# install necessary python packages
RUN pip install \
    numpy==1.13.1 \
    setuptools==27.2.0 \
    jupyter \
    pyparsing==2.2.0

# Install protobuf with python package
RUN git clone $PROTOBUF_GIT \
    && cd protobuf \
    && ./autogen.sh \
    && ./configure \
    && make -j \
    && make install \
    && ldconfig \
    && cd python \
    && python setup.py build \
    && python setup.py install

# Install ninja
RUN git clone $NINJA_GIT \
    && cd ninja \
    && python configure.py --bootstrap \
    && cp ninja /usr/local/bin

# Uncomment for python 3
# This patch will make sure cmake finds the correct python
# COPY INSTALL.patch /root/INSTALL.patch

# Clone clif 0.3 through dogan's fork
RUN git clone $CLIF_GIT \
    && cd clif \
    # && patch < /root/INSTALL.patch \ #Uncomment for python 3
    && ./INSTALL.sh $(which python) \
    && pip install .

# Install (our) kaldi
# Requires python 2.7
RUN git clone $KALDI_GIT kaldi \
    && cd kaldi/tools \
    && ./extras/check_dependencies.sh \
    && make -j4 \
    && cd ../src \
    && ./configure --shared \
    && make clean -j \
    && make depend -j \
    && make -j4

# # set env variables
ENV PATH "$PATH:/root/opt/clif/bin"
ENV KALDI_DIR "/root/kaldi"
ENV PYCLIF "/root/opt/clif/bin/pyclif"
ENV LD_LIBRARY_PATH "/root/protobuf/lib:${LD_LIBRARY_PATH}"
ENV CLIF_CXX_FLAGS "-I$CLIF_DIR/clang/lib/clang/5.0.0/include"
ENV DEBUG 1

# Download pykaldi from git and install
RUN git clone $PYKALDI pykaldi \
    && cd pykaldi \
    && python setup.py install

# # Add run permission to startup script
RUN chmod +x pykaldi/docker/start_jupyter_notebook.sh
RUN chmod +x pykaldi/docker/update_kaldi.sh
RUN chmod +x pykaldi/docker/update_pykaldi.sh

# # Define the starting command for this container and expose its running port
CMD sh -c '/root/pykaldi/docker/start_jupyter_notebook.sh 9000'
EXPOSE 9000
