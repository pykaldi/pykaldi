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
	cmake -DPYCLIF=~/opt/clif/bin/pyclif -DCMAKE_CXX_FLAGS="" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
	make kaldi-vector
```
