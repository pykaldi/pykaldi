# pyKaldi
A native-code wrapper for Kaldi in python.

## Installation

### Installing via Docker
Take the following steps to install pykaldi through Docker:
1. Install Docker in your machine as described in [Docker Documentation](https://docs.docker.com/engine/installation/)

2. Since the pykaldi Dockerfile downloads all the necessary code dependencies, including pykaldi itself, only the Dockerfile is needed. You can directly download the latest [Dockerfile here](https://github.com/usc-sail/pykaldi/blob/master/Dockerfile)

3. In order to download the private repositories needed for installation, you will need to provide your github username and password as arguments to the docker build.

```
	$ sudo docker build --tag pykaldi --build-arg githubuser=XXX --build-arg githubpasswd=XXX .
```

4. After the installation is completed, you can run an interactive version of the container
```
	$ sudo docker run -it pykaldi
```

### Installing via Source Files
TODO...