# Dockerfile for building pykaldi image from pykaldi-min image
FROM pykaldi/prereqs:latest

# Copy pykaldi directory into the container
COPY . /pykaldi

# Install PyKaldi
RUN cd /pykaldi/tools \
	&& ln -s /kaldi . \
	&& cd /pykaldi \
    && python setup.py build \
    && python setup.py install \
    && python setup.py test \
    && find kaldi -name "*.so" -delete \
    && rm -rf build
