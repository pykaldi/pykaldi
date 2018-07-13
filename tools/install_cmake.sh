#!/bin/bash

# Installation script for cmake in centos


# Checking if cmake is already higher than the required version 3.5
CV=$(cmake --version | head -1 | cut -f3 -d\ ); CV=(${CV//./ })
if (( CV[0] > 3 || CV[0] == 3 && CV[1] >= 5 )); then
    echo "Latest CMake already installed"
    exit 0
fi

yum install cmake3
echo "CMake3 installed"

echo "linking cmake3 to cmake and changing cmake to cmake_old"

sudo mv /usr/bin/cmake /usr/bin/cmake_old
echo "Old cmake renamed"

sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
echo "cmake3 successfully linked as cmake"
