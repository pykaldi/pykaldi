#!/bin/bash
# Pre-Installation checks
#   * Check that Python executable exists
#   * Check that Pip exists
#   * Checks for command dependencies
#   * Checks for libtool, zlib
#   * Checks that the python environment has numpy, setuptools and pyparsing
# Sets exit code accordingly
# 
# 
# This codes takes mostly from Kaldi check_dependencies.sh,
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
################################################################################################
set -x -e 
CXX=${CXX:-g++}
status=0

if ! which python2.7 >&/dev/null; then
  echo ""
  echo "$0: python2.7 is not installed"
  echo ""
fi

if ! which python3 >&/dev/null; then
  echo ""
  echo "$0: python3 is not installed"
  echo ""
fi


# Check (command) dependencies
for c in git cmake autoconf automake curl make g++ unzip wget svn pkg-config
do
    command -v $c >/dev/null 2>&1 || { echo >&2 "$c is required but it was not found"; }
done

command -v libtoolize >/dev/null 2>&1 || { echo "libtool is not installed"; }

# Taken from Kaldi extras/check_dependencies.sh
if ! echo "#include <zlib.h>" | $CXX -E - >&/dev/null; then
    echo ""
    echo "zlib is not installed."
    echo ""
    status=1
fi

# TODO: Check build-essential, libatlas3-base


#######################################################################################################
# Python settings
# Help cmake find the correct python
#######################################################################################################
PYTHON=$(which python)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    PYTHON="$PYTHON_EXECUTABLE"
fi

####################################################################
# Sets CLIF_DIR to be the same as the virtualenv we're
# currently running inside. 
####################################################################
CLIF_DIR=$($PYTHON -c 'from distutils.sysconfig import get_config_var; print(get_config_var("prefix"))')
if [ -z "$CLIF_DIR" ]; then
    echo ""
    echo "Python virtual environment $CLIF_DIR was not found!"
    echo ""
    status=1
fi

# Check python packages
if ! $PYTHON -c 'import numpy'; then
    echo ""
    echo "Python package numpy not found in environment."
    echo "Please install it with 'pip install \"numpy>=1.13.1\"'"
    echo ""
    status=1
else
    NV=$($PYTHON -c 'import numpy; print(numpy.__version__)' | cut -f2 -d\ ); NV=(${NV//./ })
    if (( NV[0] < 1 || NV[0] == 1 && NV[1] < 13 || NV[0] == 1 && NV[1] == 13 && NV[2] < 1 )); then
        echo ""
        echo "Numpy version ${NV[@]} found but >= 1.13.1 needed."
        echo ""
        status=1
    fi
fi

if ! $PYTHON -c 'import setuptools'; then
    echo ""
    echo "Python package setuptools not found in environment."
    echo "Please install it with 'pip install \"setuptools>=27.2.0\"'"
    echo ""
    status=1
fi

if ! $PYTHON -c 'import pyparsing'; then
    echo ""
    echo "Python package pyparsing not found in environment."
    echo "Please install it with 'pip install \"pyparsing>=2.2.0\"'"
    echo ""
    status=1
fi

####################################################################
# Check write access to package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo ""
    echo "We cannot write to $PYTHON_PACKAGE_DIR."
    echo "Did you forget runing this with sudo? "
    echo "sudo $0"
    status=1
fi

if [ $status -eq 0 ]; then
    echo "$0: all OK."
fi

exit $status
