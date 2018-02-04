#!/bin/bash

# 
# 
# Installation script for Ninja
#	Checks for previous installation of ninja 
# 	If found, then does nothing and exits
# 	If not, installs it from git
# 
set -x -e

# Check if ninja is already installed
if which ninja; then
	exit 0
fi

NINJA_DIR="$1"
NINJA_GIT="https://github.com/ninja-build/ninja.git"

if [[ "$1" =~ ^-?-h ]]; then
    echo "Usage: $0 [NINJA_DIR]"
    exit 1
fi

# Install ninja
echo "Installing ninja..."
git clone $NINJA_GIT $NINJA_DIR
cd "$NINJA_DIR"
$PYTHON_EXECUTABLE configure.py --bootstrap

echo "Done installing ninja..."