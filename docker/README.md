# Build pykaldi image

In order to download all the private repositories needed for installation, you will need to provide your github credentials. Run the following command in this directory (docker):

```
docker build --tag pykaldi --build-arg githubuser=XXX --build-arg githubpasswd=XXX .
```

# Run ipython notebook with installed pykaldi

```
docker run -p 9000:9000 pykaldi
```

# Run interactive bash mode

```
docker run -it pykaldi /bin/bash
```

