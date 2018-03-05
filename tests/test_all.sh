#!/bin/bash
set -e

for f in $(find . -name "*-test.py*"); do
    echo "Running $f"
    python $f
done
