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
#
#
# Usage: ./check_dependencies.sh [PYTHON]
#
#   PYTHON is the python executable to use (optional, defaults to current python)
#
# TODO (VM):
#   Check pip matches python
################################################################################################

set -e
CXX=${CXX:-g++}
status=0

# Which packages to check
PKGS=( git cmake autoconf automake curl make g++ unzip wget svn pkg-config )

# Which python packages to check
PY_PKGS=( numpy setuptools pyparsing )

################################################################################################
# Checks python binaries in system installation
################################################################################################
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

################################################################################################
# Check (command) dependencies
################################################################################################
for c in ${PKGS[@]}; do
    command -v $c >/dev/null 2>&1 || { echo >&2 "$c is required but it was not found"; }
done


################################################################################################
# Taken from Kaldi extras/check_dependencies.sh
# Check zlib is installed
################################################################################################
if ! echo "#include <zlib.h>" | $CXX -E - >&/dev/null; then
    echo ""
    echo "zlib is not installed."
    echo ""
    status=1
fi

if ! which libtoolize >&/dev/null && ! which glibtoolize >&/dev/null; then
  echo "$0: neither libtoolize nor glibtoolize is installed"
  add_packages libtool libtool libtool
fi

# TODO: Check build-essential, libatlas3-base

#######################################################################################################
# Define python executable to use
#######################################################################################################
PYTHON=$(which python)
if [ -n "$1" ]; then
    PYTHON="$1"
fi

####################################################################
# Checks python packages
####################################################################
for c in ${PY_PKGS[@]}; do
    $PYTHON -c "import $c"
    if [ ! $? ]; then
        echo ""
        echo "Python package $c not found in environment."
        echo ""
        status=1
    fi
done

# Checks numpy version
$PYTHON -c "import numpy"
if [ ! $? ]; then
    NV=$($PYTHON -c 'import numpy; print(numpy.__version__)' | cut -f2 -d\ ); NV=(${NV//./ })
    if (( NV[0] < 1 || NV[0] == 1 && NV[1] < 13 || NV[0] == 1 && NV[1] == 13 && NV[2] < 1 )); then
        echo ""
        echo "Numpy version ${NV[@]} found but >= 1.13.1 needed."
        echo ""
        status=1
    fi
fi

####################################################################
# Finalize
####################################################################
if [ $status -eq 0 ]; then
    echo "$0: all OK."
fi

exit $status
