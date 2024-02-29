# acts-aas

This repository contains information on how to build ACTS with the triton server and how to run inference-as-a-service (AAS) with ACTS.

## Instruction

### Build the backend
``` bash
# Assuming <work-dir> includes acts/ and acts-aas/
shifter --module=gpu --volume=<work-dir>:/workspace/ --workdir=/workspace/ --image=hrzhao076/acts-triton-dev:v0.1 /bin/bash

# or using podman-hpc
podman-hpc run -it --rm --shm-size=20G --gpu -p8000:8000 -p8001:8001 -p8002:8002 -v <work-dir>:/workspace/ -v /global/cfs/projectdirs/m3443/data/ACTS-aaS/:/global/cfs/projectdirs/m3443/data/ACTS-aaS/ hrzhao076/acts-triton-dev:v0.1 /bin/bash

cd /workspace/acts-aas
# git chekcout dev/backend

# change the env vars in Scripts/setup_env.cfg for customization
source Scripts/setup_env.cfg

# Minimal compilation of acts
./Scripts/make_acts.sh
./Scripts/make_triton.sh

# start the triton inference server
tritonserver --model-repository=$INSTALLDIR/model_repo
```
### Production library
The library has been compiled and installed at `/global/cfs/projectdirs/m3443/data/ACTS-aaS/sw/prod/ver_02012024`. For utilizing a prebuilt version:

``` bash
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/ACTS-aaS/sw/prod/ver_02012024
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH

tritonserver --model-repository=$INSTALLDIR/model_repo/
```
