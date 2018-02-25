
Pykaldi image on dockerhub is automatically rebuilt with every commit to the master branch. Udating Pykaldi image is as simple as pulling the latest from dockerhub.

```
docker login
docker pull pykaldi/pykaldi
```

Alternatively, you can build the docker image as follows:

# Building pykaldi Docker image

Run the following command in this directory (docker):

```
docker build --tag pykaldi .
```

The built image comes with [jupyter](http://jupyter.org/) notebook built in. To run it,

```
docker run -p 9000:9000 pykaldi
```

Or run in interactive bash mode

```
docker run -it pykaldi /bin/bash
```

# Updating Pykaldi or Kaldi

For your convenience, we provide two commands to update Pykaldi or Kaldi to their most recent versions. 

### For Pykaldi

```
docker run pykaldi /root/pykaldi/docker/update_pykaldi.sh
```

### For Kaldi

```
docker run pykaldi /root/pykaldi/docker/update_kaldi.sh
```


# Build pykaldi-deps
While it is not necessary for building the Docker image, pykaldi-deps allows faster testing on Travis CI. Docker does not allow for copying outside the current context, so in order to copy the tools directory into the container, we send the parent directory as context. 

```
docker build --tag pykaldi-deps -f ./Dockerfile.deps ..
```