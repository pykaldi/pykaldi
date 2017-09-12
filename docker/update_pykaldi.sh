#!/bin/bash

PYTHON=python3

cd pykaldi \
&& git pull origin master \
&& $PYTHON setup.py install \
&& echo "DONE"
