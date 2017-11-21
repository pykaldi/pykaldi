#!/bin/bash

PYTHON=python

cd pykaldi \
&& git pull origin master \
&& $PYTHON setup.py install \
&& echo "DONE"
