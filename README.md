# pyKaldi
A native-code wrapper for Kaldi in python.

# Build instructions
```
	virtualenv /tmp/pykaldienv
	cmake -DPYCLIF=/home/victor/opt/clif/bin/pyclif -DCMAKE_CXX_FLAGS="-I/usr/include/ -L/usr/include -I/usr/include/python2.7"
	
	make kaldi-utils
	/tmp/pykaldienv/bin/pip install .
	/tmp/pykaldienv/bin/python pyKaldi_test
```