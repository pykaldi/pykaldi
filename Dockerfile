FROM ubuntu:latest

WORKDIR /root

# update ubuntu and install essentials
RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake autoconf \
														  automake libtool \
														  curl make g++ unzip \
														  python-dev build-essential \
														  python-pip virtualenv \
														  libatlas3-base wget \
														  zlib1g-dev subversion \
														  pkg-config

# install necessary python packages
RUN pip install numpy==1.13.1 setuptools==27.2.0 

# Add pykaldi to the container
ADD . /root/pykaldi

# Update this with your own credentials
# at time of build
ARG githubuser
ARG githubpasswd

# Install (our) kaldi
RUN git clone https://$githubuser:$githubpasswd@github.com/usc-sail/kaldi-pykaldi.git kaldi && \
	cd kaldi/tools && \
	./extras/check_dependencies.sh && \
	make -j4 && \
	cd ../src && \
	./configure --shared && \
	make clean -j && \
	make depend -j && \
	make -j4

# Install protobuf with python package
RUN git clone https://github.com/google/protobuf.git && \
	cd protobuf && \
	./autogen.sh && \
	./configure && \
	make -j && \
	make install && \
	ldconfig && \
	cd python && \
	python setup.py build && \
	python setup.py install

# Install ninja
RUN git clone https://github.com/ninja-build/ninja.git && \
	cd ninja && \
	./configure.py --bootstrap && \
	cp ninja /usr/local/bin

# Install clif (apply patch so that install does not use virtual env)
RUN git clone https://github.com/google/clif.git && \
	cd clif && \
	patch < /root/pykaldi/extras/clif/install.diff && \
	./INSTALL.sh $(which python)

# set env variables
ENV KALDI_DIR /root/kaldi/
ENV CLIF_CXX_FLAGS "-I/usr/lib/gcc/x86_74-linux-gnu/5/include-fixed -I/usr/include"
ENV CLIF_DIR /root/opt/clif
ENV DEBUG "TRUE"
ENV PYCLIF "/root/opt/clif/bin/pyclif"

RUN echo "$CLIF_DIR"
RUN echo "$KALDI_DIR"
RUN echo "$PYCLIF"

# install pykaldi
RUN python /root/pykaldi/setup.py install
