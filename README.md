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
	#Activate clif virtualenv
	source opt/clif/bin/activate
	
	# Change directory and configure
	cd pykaldi
	cmake -DPYCLIF=~/opt/clif/bin/pyclif -DCMAKE_CXX_FLAGS="-I/home/victor/clif_backend/build_matcher/lib/clang/5.0.0/include -I/home/victor/Workspace/pykaldi/kaldi/src -I/home/victor/Workspace/kaldi/tools/openfst/include -I/home/victor/Workspace/kaldi/tools/ATLAS/include -std=c++11" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
	make kaldi-vector
```
