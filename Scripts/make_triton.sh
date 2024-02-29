#!/bin/bash
SCRIPTDIR="$(dirname "$(realpath "$0")")"
source $SCRIPTDIR/setup_env.cfg

SOURCEDIRTRITON=$SCRIPTDIR/../ActsExaTrkXTritonBackend
cmake -B $BUILDDIRTRITON -S $SOURCEDIRTRITON \
-DCMAKE_PREFIX_PATH=$INSTALLDIR \
-DCMAKE_INSTALL_PREFIX=$INSTALLDIR

cmake --build $BUILDDIRTRITON --target install -- -j20
