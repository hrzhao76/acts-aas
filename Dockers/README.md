# Docker images 

Currently we have two docker images: 
1. `hrzhao076/acts-triton-dev:c2e21bd6` , base image: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
2. `hrzhao076/acts-triton-dev:v0.1` base image: nvcr.io/nvidia/tritonserver:22.02-py3

The recommended one is the 2nd image, although some warning appears.  
``` bash
shifter --module=gpu --volume=/pscratch/sd/h/hrzhao/Projects/:/workspace/ --workdir=/workspace/ --image=hrzhao076/acts-triton-dev:v0.1 /bin/bash
```

# Dev Notes 
## How the docker image is built 
Currently due to the `podman-hpc` inavailability on Perlmutter, the docker image is built using the CI/CD pipeline on `nrp-nautilus`. Then the image is stored in the `nrp-nautilus` gitlab registry. A manual step is required to pull the image from the registry and push it to the `dockerhub` registry for the image to be available on Perlmutter, e.g. `shifterimg -v pull hrzhao076/acts-triton-dev:c2e21bd6`. 

## How to run the docker image 
``` bash
shifter --module=gpu --volume=/pscratch/sd/h/hrzhao/Projects/:/workspace/ --workdir=/workspace/ --image=hrzhao076/acts-triton-dev:c2e21bd6 /bin/bash

# The following error occurs with the raw base.. 
# Error is as follows if the volume folder does not exist(idk why): 
# FAILED to create volume "to": /var/udiMount//workspace/, cannot create mount points in that location
# FAILED to setup user-requested mounts.
# FAILED to setup image.

shifter -v --module=gpu --volume=/pscratch/sd/h/hrzhao/Projects/acts/:/home/ --workdir=/home/ --image=hrzhao076/acts-triton-dev:9400c6d5 /bin/bash

```

## history of docker imqge
1. d1288b7f: only triton client to debug 
2. 26f4ec5b: added the triton server; no python backend as the boost sha error 
3. 9400c6d5: rmove the pkgs to reduce the image size; mkdir -p /workspace 
4. c2e21bd6: add /workspace/testfile to keep /workspace/ in the image(I don't know why /workspace disappaers in the image otherwise)