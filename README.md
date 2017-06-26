# pyKaldi
A native-code wrapper for Kaldi in python.

## Installation

### Prerequisites

 1. [CMake](http://llvm.org/docs/CMake.html) version 3.5 or later is available.

 2. Google [protobuf](https://developers.google.com/protocol-buffers/docs/downloads)
    for inter-process communication between the CLIF frontend and backend.
    Version 3.2.0 or later is required.
    Please install protobuf for both C++ and Python from source, as we will
    need some protobuf source code later on.

 3. You must have [virtualenv](https://pypi.python.org/pypi/virtualenv)
    installed.
    
 4. Optional: Install [Ninja](https://github.com/ninja-build/ninja)
    
 4. Install [Clif](https://github.com/google/clif/)
    
# Build instructions
```
	# Define CXX_FLAGS:
	export CXX_FLAGS=-I/usr/lib/gcc/x86_64-linux-gnu/5/include-fixed -I/home/victor/clif_backend/build_matcher/lib/clang/5.0.0/include -I/usr/include

	# Activate pyclif environment
	source ~/opt/clif/bin/activate

	# Provide a kaldi directory
	export KALDI_DIR="/saildisk/tools/Kaldi/"

	# Do python setup
	DEBUG=1 python setup.py build

	# Install
	python setup.py install
```
