#!/bin/bash
SCRIPTDIR="$(dirname "$(realpath "$0")")"
source $SCRIPTDIR/setup_env.cfg

cmake -B $BUILDDIRACTS -S $SOURCEDIRACTS \
    -DACTS_BUILD_PLUGIN_EXATRKX=ON \
    -DACTS_BUILD_EXAMPLES_EXATRKX=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$INSTALLDIR

cmake --build $BUILDDIRACTS --target install -- -j20

export DD4hepINSTALL=/usr/local/

source ${INSTALLDIR}/bin/this_acts.sh
