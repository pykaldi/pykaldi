#!/bin/bash
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installation script for CLIF.
# Adapted from https://github.com/google/clif/blob/master/INSTALL.sh
#
# Usage:
#   ./install_clif.sh [PYTHON_EXECUTABLE] [PYTHON_LIBRARY]
#
#   PYTHON_EXECUTABLE - python executable to use (default: python)
#   PYTHON_LIBRARY - overrides the python library to use

set -e

TOOLS_DIR="$PWD"
CLIF_DIR="$PWD/clif"

PYTHON="python"
if [ -n "$1" ]; then
  PYTHON="$1"
fi
PYTHON_EXECUTABLE="$(which $PYTHON)"

PYTHON_LIBRARY=$($PYTHON_EXECUTABLE $TOOLS_DIR/find_python_library.py)
if [ -n "$2" ]; then
  PYTHON_LIBRARY="$2"
fi
if [ ! -f "$PYTHON_LIBRARY" ]; then
  echo "Python library $PYTHON_LIBRARY could not be found."
  echo "Please specify the python library as an argument to $0"
  echo "e.g. $0 /usr/bin/python3 /usr/lib/x86_64-linux-gnu/libpython3.5m.so.1"
  exit 1
fi

PYTHON_PIP="$PYTHON_EXECUTABLE -m pip"
PYTHON_ENV=$($PYTHON_EXECUTABLE -c "import sys; print(sys.prefix)")
PYTHON_INCLUDE_DIR=$($PYTHON_EXECUTABLE -c 'from sysconfig import get_paths; print(get_paths()["include"])')
CMAKE_PY_FLAGS=(-DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" -DPYTHON_LIBRARY="$PYTHON_LIBRARY")

####################################################################
# Ensure CMake is installed (needs 3.5+)
####################################################################
CV=$(cmake --version | head -1 | cut -f3 -d\ ); CV=(${CV//./ })
if (( CV[0] < 3 || CV[0] == 3 && CV[1] < 5 )); then
  echo "Install CMake version 3.5+"
  exit 1
fi

####################################################################
# Ensure Google protobuf C++ library is installed (needs v3.2+).
####################################################################
if [ -d "$TOOLS_DIR/protobuf" ]; then
  export PATH="$TOOLS_DIR/protobuf/bin:$PATH"
  export PKG_CONFIG_PATH="$TOOLS_DIR/protobuf:$PKG_CONFIG_PATH"
fi
PV=$(protoc --version | cut -f2 -d\ ); PV=(${PV//./ })
if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
  echo "Install Google protobuf version 3.2+"
  exit 1
fi
PROTOBUF_PREFIX_PATH="$(dirname "$(dirname "$(which protoc)")")"
PROTOBUF_INCLUDE="$(pkg-config --cflags protobuf)"
PROTOBUF_LIBS="$(pkg-config --libs protobuf)"

######################################################################

CLIF_GIT="-b pykaldi https://github.com/pykaldi/clif.git"
LLVM_DIR="$CLIF_DIR/../clif_backend"
BUILD_DIR="$LLVM_DIR/build_matcher"

if [ ! -d "$CLIF_DIR" ]; then
  git clone $CLIF_GIT $CLIF_DIR
else
  echo "Destination $CLIF_DIR already exists."
fi

cd "$CLIF_DIR"

declare -a CMAKE_G_FLAG
declare -a MAKE_PARALLELISM
if which ninja; then
  CMAKE_G_FLAGS=(-G Ninja)
  MAKE_OR_NINJA="ninja"
  MAKE_PARALLELISM=()  # Ninja does this on its own.
  # ninja can run a dozen huge ld processes at once during install without
  # this flag... grinding a 12 core workstation with "only" 32GiB to a halt.
  # linking and installing should be I/O bound anyways.
  MAKE_INSTALL_PARALLELISM=(-j 2)
  echo "Using ninja for the clif backend build."
else
  CMAKE_G_FLAGS=()  # The default generates a Makefile.
  MAKE_OR_NINJA="make"
  MAKE_PARALLELISM=(-j 4)
  if [[ -r /proc/cpuinfo ]]; then
    N_CPUS="$(cat /proc/cpuinfo | grep -c ^processor)"
    [[ "$N_CPUS" -gt 0 ]] && MAKE_PARALLELISM=(-j $N_CPUS)
    MAKE_INSTALL_PARALLELISM=(${MAKE_PARALLELISM[@]})
  fi
  echo "Using make.  Build will take a long time.  Consider installing ninja."
fi

# Download, build and install LLVM and Clang (needs a specific revision).

mkdir -p "$LLVM_DIR"
cd "$LLVM_DIR"
svn co https://llvm.org/svn/llvm-project/llvm/trunk@307315 llvm
cd llvm/tools
svn co https://llvm.org/svn/llvm-project/cfe/trunk@307315 clang
ln -s -f -n "$CLIF_DIR/clif" clif

# Build and install the CLIF backend.  Our backend is part of the llvm build.
# NOTE: To speed up, we build only for X86. If you need it for a different
# arch, change it to your arch, or just remove the =X86 line below.

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_INSTALL_PREFIX="$PYTHON_ENV/clang" \
      -DCMAKE_PREFIX_PATH="$PROTOBUF_PREFIX_PATH" \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=true \
      -DLLVM_INSTALL_TOOLCHAIN_ONLY=true \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_BUILD_DOCS=false \
      -DLLVM_TARGETS_TO_BUILD=X86 \
      "${CMAKE_PY_FLAGS[@]}" \
      "${CMAKE_G_FLAGS[@]}" "$LLVM_DIR/llvm"
"$MAKE_OR_NINJA" "${MAKE_PARALLELISM[@]}" clif-matcher clif_python_utils_proto_util
"$MAKE_OR_NINJA" "${MAKE_INSTALL_PARALLELISM[@]}" install

# Get back to the CLIF Python directory and have pip run setup.py.

cd "$CLIF_DIR"
# Grab the python compiled .proto
cp "$BUILD_DIR/tools/clif/protos/ast_pb2.py" clif/protos/
# Grab CLIF generated wrapper implementation for proto_util.
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.cc" clif/python/utils/
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.h" clif/python/utils/
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.init.cc" clif/python/utils/

####################################################################
# Check write access to Python package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
  echo ""
  echo "*** PYTHON_PACKAGE_DIR=$PYTHON_PACKAGE_DIR"
  echo "*** Writing to PYTHON_PACKAGE_DIR requires sudo access."
  echo "*** Run the following command to install pyclif Python package."
  echo ""
  echo "sudo CFLAGS=\"$PROTOBUF_INCLUDE\" LDFLAGS=\"$PROTOBUF_LIBS\" $PYTHON_PIP install $CLIF_DIR"
  exit 1
else
  CFLAGS="$PROTOBUF_INCLUDE" LDFLAGS="$PROTOBUF_LIBS" $PYTHON_PIP install .
fi

echo "Done installing CLIF."
