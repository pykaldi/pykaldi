#!/bin/bash

################################################################################################
# Pre-Installation checks
#   * Check that Python executable exists
#   * Check that Pip exists
#   * Checks for command dependencies
#   * Checks for libtool, zlib
#   * Checks that the python environment has numpy, setuptools and pyparsing
# Sets exit code accordingly
################################################################################################
CXX=${CXX:-g++}
status=0

# Check if python is installed
if ! command -v python >/dev/null 2>&1; then
    echo "$PYTHON_EXECUTABLE command not found!"
    status=1
fi

# Check if pip is installed
if ! command -v pip >/dev/null 2>&1; then
    echo "$PYTHON_PIP command not found!"
    status=1
fi

# Check (command) dependencies
for c in git cmake autoconf automake curl make g++ unzip wget svn pkg-config
do
    command -v $c >/dev/null 2>&1 || { echo >&2 "$c is required but it was not found"; }
done

command -v libtoolize >/dev/null 2>&1 || { echo "libtool is not installed"; }


# Taken from Kaldi extras/check_dependencies.sh
if ! echo "#include <zlib.h>" | $CXX -E - >&/dev/null; then
    echo "zlib is not installed."
    status=1
fi

# TODO: Check build-essential, libatlas3-base

# Check python packages
echo ""
echo ""

if ! $PYTHON_EXECUTABLE -c 'import numpy'; then
    echo "Python package numpy not found in environment."
    echo "Please install it with 'pip install \"numpy>=1.13.1\"'"
    status=1
else
    NV=$(python -c 'import numpy; print(numpy.__version__)' | cut -f2 -d\ ); NV=(${NV//./ })
    if (( NV[0] < 1 || NV[0] == 1 && NV[1] < 13 || NV[0] == 1 && NV[1] == 13 && NV[2] < 1 )); then
        echo "Numpy version ${NV[@]} found but >= 1.13.1 needed."
        status=1
    fi
fi

echo ""
echo ""

if ! $PYTHON_EXECUTABLE -c 'import setuptools'; then
    echo "Python package setuptools not found in environment."
    echo "Please install it with 'pip install \"setuptools>=27.2.0\"'"
    status=1
fi

echo ""
echo ""

if ! $PYTHON_EXECUTABLE -c 'import pyparsing'; then
    echo "Python package pyparsing not found in environment."
    echo "Please install it with 'pip install \"pyparsing>=2.2.0\"'"
    status=1
fi

if [ $status -eq 0 ]; then
    echo "$0: all OK."
fi

exit $status
