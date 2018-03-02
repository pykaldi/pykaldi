#!/bin/bash

for f in $(find . -name "*.py*"); do
    python $f;
done
