#!/bin/bash

python compute-vad.py --test-plot scp:data/wav.scp ark,t:out/ltsv-feats.ark
