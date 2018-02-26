# Installing Pykaldi Image

PyKaldi image on Docker Hub is automatically updated with every commit to the
master branch. Installing and updating PyKaldi image is as simple as pulling the
latest image from Docker Hub.

```bash
docker pull pykaldi/pykaldi
```

# Building Pykaldi Image

If you would like to build PyKaldi image yourself instead of downloading it from
Docker Hub, run the following command inside the `pykaldi/docker` directory.
Note the `..` at the end. To copy the contents of `pykaldi` directory into the
container, we set the parent directory, i.e. `pykaldi`, as the build context.

```bash
docker build --tag pykaldi/pykaldi -f Dockerfile ..
```

# Running PyKaldi Image

## Bash

You can run PyKaldi image in interactive mode with the following command.

```bash
docker run -it pykaldi/pykaldi /bin/bash
```

## Jupyter Notebook

PyKaldi image comes with IPython and Jupyter. You can use PyKaldi inside a
[Jupyter](http://jupyter.org/) notebook by first starting the server

```bash
docker run -it -p 9000:9000 pykaldi/pykaldi /bin/bash
jupyter notebook --no-browser --ip=* --port=9000 --allow-root
```

and then navigating to [http://localhost:9000](http://localhost:9000) using
your favorite web browser.
