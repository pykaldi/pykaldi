# Build pykaldi image

Run the following command in this directory (docker):

```
docker build --tag pykaldi .
```

# Run ipython notebook with installed pykaldi

```
docker run -p 9000:9000 pykaldi
```

# Run interactive bash mode

```
docker run -it pykaldi /bin/bash
```

# Updating pykaldi or kaldi

For your convenience, we provide two commands to update pykaldi or kaldi to their most recent versions. 

### For Pykaldi

```
docker run pykaldi /root/pykaldi/docker/update_pykaldi.sh
```

### For Kaldi

```
docker run pykaldi /root/pykaldi/docker/update_kaldi.sh
```
