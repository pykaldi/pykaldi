Installation
************

Via Docker
##########

Installing through Docker can be done with either of the next two options:

* Downloading the binary from dockerhub

* Building docker image from source files

1. Downloading the binary
=========================

#. Login to dockerhub

    >>> docker login

#. Download the pre-built binary

    >>> docker pull vrmpx/pykaldi

2. Building docker from source
==============================

Take the following steps to install PyKaldi through Docker:

#. Install Docker in your machine as described in
   `Docker Documentation <https://docs.docker.com/engine/installation/>`_

#. In order to download the private repositories needed for installation, you
   will need to provide your github username and password as arguments to the
   docker build.

   >>> sudo docker build --tag pykaldi .

#. After the installation is completed, you can run an interactive version of
   the container

   >>> sudo docker run -it pykaldi

Via Source Files
################

1. Follow the installation instructions for `Protobuf
   <https://github.com/google/protobuf.git>`__ C++ and Python package.

   >>> sudo apt-get install autoconf automake libtool curl make g++ unzip
   >>> git clone https://github.com/google/protobuf.git protobuf
   >>> cd protobuf
   >>> ./autogen.sh
   >>> ./configure && make -j4
   >>> sudo make install
   >>> sudo ldconfig
   >>> cd python
   >>> python setup.py build
   >>> python setup.py install

2. We use a fork of CLIF that supports documentation within the CLIF file.
   The source code can be found `here <https://github.com/dogancan/clif/tree/pykaldi>`_.
   Clone this repository, making sure to checkout the pykaldi branch.
   Run the following commands to install the correct version:

   >>> cd
   >>> git clone -b pykaldi https://github.com/dogancan/clif/
   >>> cd clif
   >>> ./INSTALL $(which python)

   Note that if there is more than one Python version installed (e.g., Python
   2.7 and 3.6) cmake may not be able to find the correct python libraries. To
   help cmake use the correct Python, add the following options to the cmake
   command inside INSTALL.sh (make sure to substitute the correct path for your
   system):

   >>> cmake ... \
       -DPYTHON_INCLUDE_DIR="/usr/include/python3.6" \
       -DPYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.6m.so" \
       -DPYTHON_EXECUTABLE="/usr/bin/python3.6" \
       "${CMAKE_G_FLAGS[@]}" "$LLVM_DIR/llvm"

3. Download and install this fork from
   `Kaldi <https://github.com/usc-sail/kaldi-pykaldi.git>`_ which contains
   modifications necessary for CLIF compatibility.

   >>> cd
   >>> git clone https://github.com/usc-sail/kaldi-pykaldi.git kaldi
   >>> cd kaldi/tools
   >>> ./extras/check_dependencies.sh && make -j4
   >>> cd ../src
   >>> ./configure --shared && make clean -j4 && make depend -j4 && make -j4

4. Set the following environmental variables, make sure to replace the correct
   values for your installation directories.

   >>> export KALDI_DIR=<directory where kaldi was installed>
   >>> export CLIF_DIR=<directory where clif was installed>
   >>>
   >>> #optional
   >>> export DEBUG=1
   >>> export PYCLIF=<pyclif executable location>

5. Download and install `PyKaldi <https://github.com/usc-sail/pykaldi/>`_.

   >>> cd
   >>> git clone https://github.com/usc-sail/pykaldi/ pykaldi
   >>> cd pykaldi
   >>> python setup.py install
