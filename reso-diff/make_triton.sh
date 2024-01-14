export SOURCEDIR=/workspace/exatrkx-service/
export BUILDDIR=/workspace/acts-aas/reso-diff/build_triton
export INSTALLDIR=${1-/workspace/acts-aas/reso-diff/install}
export RUNDIR=/workspace/acts-aas/reso-diff/run

export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH

cd $SOURCEDIR/exatrkx_gpu
rm -rf build/
./make.sh -j 8
