export SOURCEDIR=/workspace/acts
export BUILDDIR=/workspace/acts-aas/reso-diff/build_acts
export INSTALLDIR=${1-/workspace/acts-aas/reso-diff/install}
export RUNDIR=/workspace/acts-aas/reso-diff/run

cmake -B $BUILDDIR -S $SOURCEDIR \
    -DACTS_BUILD_EXAMPLES=ON \
    -DACTS_BUILD_UNITTESTS=OFF \
    -DACTS_BUILD_INTEGRATIONTESTS=ON \
    -DACTS_BUILD_BENCHMARKS=ON \
    -DACTS_BUILD_FATRAS=ON \
    -DACTS_BUILD_DOCS=OFF \
    -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -DACTS_BUILD_EXAMPLES_HEPMC3=ON \
    -DACTS_BUILD_PLUGIN_EXATRKX=ON \
    -DACTS_BUILD_EXAMPLES_EXATRKX=ON \
    -DACTS_BUILD_PLUGIN_DD4HEP=ON \
    -DACTS_BUILD_EXAMPLES_DD4HEP=ON \
    -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/usr/local/share/cmake/TorchScatter \
    -DACTS_BUILD_ODD=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} 
    # -DACTS_BUILD_EXAMPLES_BINARIES=ON 

cmake --build $BUILDDIR --target install -- -j20  
export ACTS_PATH="/workspace/acts/"
export DEMO_PATH="/workspace/exatrkx-acts-demonstrator/"
export ACTS_BUILD_PATH=$BUILDDIR
export ACTS_INSTALL_PATH=$INSTALLDIR
export DD4hepINSTALL=/usr/local/

source ${INSTALLDIR}/bin/this_acts.sh
source ${INSTALLDIR}/bin/this_odd.sh
source ${INSTALLDIR}/python/setup.sh 

if [ ! -d "$RUNDIR" ]; then
    mkdir -p $RUNDIR
fi
# cd $RUNDIR && echo "Switching into the run dir $RUNDIR"
