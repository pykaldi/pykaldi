#!/bin/bash

PORT=$1

jupyter notebook --no-browser --ip=* --port=$PORT --allow-root 