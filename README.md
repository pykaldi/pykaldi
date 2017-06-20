# pyKaldi
A native-code wrapper for Kaldi in python.

# Build instructions
```
	virtualenv /tmp/pykaldienv
	cmake -DPYCLIF=/home/victor/opt/clif/bin/pyclif -DCMAKE_CXX_FLAGS="-I/usr/include -I/usr/include/python2.7 -I/home/victor/Workspace/pykaldi/kaldi/src -I/home/victor/Workspace/kaldi/tools/openfst/include -I/home/victor/Workspace/kaldi/tools/ATLAS/include" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

	make kaldi-vector
	/tmp/pykaldienv/bin/pip install .
	/tmp/pykaldienv/bin/python pyKaldi_test
```
