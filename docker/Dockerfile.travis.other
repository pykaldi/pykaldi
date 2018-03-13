# Dockerfile for building pykaldi image from pykaldi-min image
FROM pykaldi/pykaldi:latest

# Copy pykaldi directory into the container
COPY . /pykaldi_travis

# Install PyKaldi
RUN cd /pykaldi \
	&& git fetch /pykaldi_travis \
	&& git checkout FETCH_HEAD \
	&& python setup.py develop --uninstall \
	&& find kaldi -name "*.so" -delete \
	&& python setup.py develop \
	&& rm -rf /pykaldi_travis
