Installation
============

Via Docker
------------

Take the following steps to install PyKaldi through Docker:

#. Install Docker in your machine as described in `Docker Documentation <https://docs.docker.com/engine/installation/>`_

#. In order to download the private repositories needed for installation, you will need to provide your github username and password as arguments to the docker build.

   >>> sudo docker build --tag pykaldi --build-arg githubuser=XXX --build-arg githubpasswd=XXX .

#. After the installation is completed, you can run an interactive version of the container

   >>> sudo docker run -it pykaldi

Via Source Files
----------------

#. Follow the installation instructions for `Protobuf <https://github.com/google/protobuf.git>`__ C++ and Python package.

   .. warning:: Following commands may be outdated.

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

#. Follow the instructions to install
   `clif <https://github.com/google/clif/>`_

   >>> cd
   >>> git clone https://github.com/google/clif.git clif
   >>> cd clif
   >>> ./INSTALL $(which python)

#. Download and install this version of `Kaldi <https://github.com/usc-sail/kaldi-pykaldi.git>`_ which contains modifications necesary for clif.

	>>> cd
	>>> git clone https://github.com/usc-sail/kaldi-pykaldi.git kaldi
	>>> cd kaldi/tools
	>>> ./extras/check_dependencies.sh && make -j4
	>>> cd ../src
	>>> ./configure --shared && make clean -j4 && make depend -j4 && make -j4

#. Set the following environmental variables, make sure to replace the correct values for your installation directories

	>>> export KALDI_DIR=<directory where kaldi was installed>
	>>> export CLIF_DIR=<directory where clif was installed>
	>>>
	>>> #optional
	>>> export DEBUG=1
	>>> export PYCLIF=<pyclif executable location>

#. Download and install `PyKaldi <https://github.com/usc-sail/pykaldi/>`_ source code (i.e., this repository)

	>>> cd
	>>> git clone https://github.com/usc-sail/pykaldi/ pykaldi
	>>> cd pykaldi
	>>> python setup.py install
