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

# Install CLIF primer script.
# NOTE (VM):
# Downloads from github
# Includes modifications needed to install on Python 3.0
# 
#
# Usage:
#   ./install_clif.sh [CLIFSRC_DIR] [CLIF_VIRTUALENV] [..CMAKE_PY_FLAGS..]
#
#   CLIFSRC_DIR - directory where clif will be installed (default: $PWD)
#   CLIF_VIRTUALENV - directory of the virtualenv where to install pyclif (default to "$CLIFSRC_DIR/../opt")
#   ..CMAKE_PY_FLAGS.. - Flags for CMAKE to find the correct python bin and libs
#   
#   Env vars:
#      $PYTHON_EXECUTABLE - path to the python binaries
#      $PYTHON_LIBRARY - path to the python libraries

set -e -x

# From: https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# Gets the CWD regardless of where this script is called from
TOOLS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CLIFSRC_DIR="$PWD"
if [[ -n "$1" ]]; then
  CLIFSRC_DIR="$1"
  shift
fi

CLIF_VIRTUALENV="$CLIFSRC_DIR/../opt"
if [[ -n "$1" ]]; then
  CLIF_VIRTUALENV="$1"
  shift
fi


#######################################################################################################
# Python settings
# Help cmake find the correct python
#######################################################################################################
#if [ ! -z "$PYTHON_EXECUTABLE" ]; then
  PYTHON_EXECUTABLE=$(which python)
#fi

  PYTHON_PIP=$(which pip)

#if [ ! -z "$PYTHON_LIBRARY" ]; then
  PYTHON_LIBRARY=$($PYTHON_EXECUTABLE $TOOLS_DIR/findPythonLib.py)
#fi




PYTHON_INCLUDE_DIR=$($PYTHON_EXECUTABLE -c 'from sysconfig import get_paths; print(get_paths()["include"])')
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

####################################################################
# Fix cmake flags to find the correct python
####################################################################
CMAKE_PY_FLAGS=(-DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" -DPYTHON_LIBRARY="$PYTHON_LIBRARY")
if [[ -n "$@" ]]; then
  CMAKE_PY_FLAGS=( "$@" )
fi

####################################################################
# Ensure CMake is installed (needs 3.5+)
####################################################################
CV=$(cmake --version | head -1 | cut -f3 -d\ ); CV=(${CV//./ })
if (( CV[0] < 3 || CV[0] == 3 && CV[1] < 5 )); then
  echo "Install CMake version 3.5+"
  exit 1
fi

####################################################################
# Ensure Google protobuf C++ source is installed (needs v3.2+).
####################################################################
PV=$(protoc --version | cut -f2 -d\ ); PV=(${PV//./ })
if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
  echo "Install Google protobuf version 3.2+"
  exit 1
fi
PROTOC_PREFIX_PATH="$(dirname "$(dirname "$(which protoc)")")"

######################################################################
# Protobuf might not be a global installation
# Find the location for the includes and libs
######################################################################
PROTOBUF_INCLUDE="$(pkg-config --cflags protobuf)"
PROTOBUF_LIBS="$(pkg-config --libs protobuf)"

######################################################################
CLIF_GIT="-b pykaldi https://github.com/pykaldi/clif.git"
LLVM_DIR="$CLIFSRC_DIR/../clif_backend"
BUILD_DIR="$LLVM_DIR/build_matcher"

if $DEBUG; then
  echo ""
  echo "Installing clif with the following params: "
  echo "PATH:$PATH"
  echo "CLIF_GIT: $CLIF_GIT"
  echo "CLIFSRC_DIR: $CLIFSRC_DIR"
  echo "CLIF_VIRTUALENV: $CLIF_VIRTUALENV"
  echo "LLVM_DIR: $LLVM_DIR"
  echo "BUILD_DIR: $BUILD_DIR"
  echo "PROTOBUF_INCLUDE: $PROTOBUF_INCLUDE"
  echo "PROTOBUF_LIBS: $PROTOBUF_LIBS"
  echo "CMAKE_PY_FLAGS: ${CMAKE_PY_FLAGS[@]}"
  echo ""
#  exit 1
fi


if [ -d "$CLIFSRC_DIR" ]; then
  echo "Destination $CLIFSRC_DIR already exists!, skipping."
  exit 0
fi


# Install clif from dogan's fork
git clone $CLIF_GIT $CLIFSRC_DIR
cd "$CLIFSRC_DIR"

# If Ninja is installed, use it instead of make.  MUCH faster.

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
  MAKE_PARALLELISM=(-j 2)
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
ln -s -f -n "$CLIFSRC_DIR/clif" clif

# Build and install the CLIF backend.  Our backend is part of the llvm build.
# NOTE: To speed up, we build only for X86. If you need it for a different
# arch, change it to your arch, or just remove the =X86 line below.

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_INSTALL_PREFIX="$CLIF_VIRTUALENV/clang" \
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

cd "$CLIFSRC_DIR"
# Grab the python compiled .proto
cp "$BUILD_DIR/tools/clif/protos/ast_pb2.py" clif/protos/
# Grab CLIF generated wrapper implementation for proto_util.
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.cc" clif/python/utils/
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.h" clif/python/utils/
cp "$BUILD_DIR/tools/clif/python/utils/proto_util.init.cc" clif/python/utils/
CFLAGS="$PROTOBUF_INCLUDE" LDFLAGS="$PROTOBUF_LIBS" "$PYTHON_PIP" install .

echo "Clif installed to $CLIF_VIRTUALENV"
touch "$CLIFSRC_DIR/.DONE"
exit 0
