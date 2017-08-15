# pyKaldi
A native-code wrapper for Kaldi in python.

## Installation

### via Docker
Take the following steps to install pykaldi through Docker:
1. Install Docker in your machine as described in [Docker Documentation](https://docs.docker.com/engine/installation/)

2. In order to download the private repositories needed for installation, you will need to provide your github username and password as arguments to the docker build.

```
	$ sudo docker build --tag pykaldi --build-arg githubuser=XXX --build-arg githubpasswd=XXX .
```

3. After the installation is completed, you can run an interactive version of the container
```
	$ sudo docker run -it pykaldi
```

### via Source Files

1. Follow the installation instructions for [Protobuf](https://github.com/google/protobuf.git) C++ and Python package.
```
	$ git clone git clone https://github.com/google/protobuf.git protobuf
	$ cd protobuf
	$ ./autogen.sh 
	$ ./configure && make -j4
	$ sudo make install
	$ sudo ldconfig
	$ cd python
	$ python setup.py build
	$ python setup.py install
```

2. Follow the instructions to install [Clif](https://github.com/google/clif/).
```
	$ git clone https://github.com/google/clif.git clif
	$ cd clif
	$ ./INSTALL $(which python)
```

3. Download and install this version of [Kaldi](https://github.com/usc-sail/kaldi-pykaldi.git) which contains modifications necesary for clif.
```
	$ git clone https://github.com/usc-sail/kaldi-pykaldi.git kaldi
	$ cd kaldi/tools
	$ ./extras/check_dependencies.sh && make -j4
	$ cd ../src
	$ ./configure --shared && make clean -j4 && make depend -j4 && make -j4
```

4. Download [PyKaldi](https://github.com/usc-sail/pykaldi/) source code (i.e., this repository)
```
	$ git clone https://github.com/usc-sail/pykaldi/ pykaldi
```

5. Set the following environmental variables, make sure to replace the correct values for the installation directories:
```
	$ export KALDI_DIR=<directory where kaldi was installed>
	$ export CLIF_CXX_FLAGS="-I/usr/lib/gcc/x86_74-linux-gnu/5/include-fixed -I/usr/include"
	$ export CLIF_DIR=<directory where clif was installed>
	$ export DEBUG="True"
	$ export PYCLIF=<pyclif binary location>
```

6. Install __pykaldi__
```
	$ python setup.py install
```
