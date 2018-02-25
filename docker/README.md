
Pykaldi image on dockerhub is automatically rebuilt with every commit to the master branch. Udating Pykaldi image is as simple as pulling the latest from dockerhub.

```
docker pull pykaldi/pykaldi
```

Alternatively, you can build the docker image.

# Building Pykaldi Docker image
Run the following command in this directory (docker). Please note the `..` sent as context. This is due to a limitation in which Docker does not allow copying items outside the current context. In order for us to copy the `tools` directory into the container, we send the parent directory as context.

```
docker build --tag pykaldi:latest -f Dockerfile ..
```

# Running PyKaldi Docker image

## Bash
You can bash into the container with the following command

```
docker run -it pykaldi /bin/bash
```

## Jupyter
The built image comes with [jupyter](http://jupyter.org/) notebook built in. To run it,

```
docker run -it -p 9000:9000 pykaldi /bin/bash -c 'jupyter notebook --no-browser --ip=* --port=9000 --allow-root'
```

And then point your favorite web browser to `localhost:9000`.

# Build pykaldi-deps
While it is not necessary for building the Docker image, pykaldi-deps allows faster testing on Travis CI. 

```
docker build --tag pykaldi-deps -f ./Dockerfile.deps ..
```