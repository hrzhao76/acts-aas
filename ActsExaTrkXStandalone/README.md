# ActsExaTrkXStandalone

## Compile ActsExaTrkXStandalone
``` bash
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/ACTS-aaS/sw/prod/ver_01192024
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH

cd /workspace/acts-aas/ActsExaTrkXStandalone
cmake -S . -B build -DCMAKE_PREFIX_PATH=$INSTALLDIR
cmake --build build -- -j 8
```

## Run ActsExaTrkXStandalone
If you haven't run the inference.py script, you need first generate spacepoint.csv file after compiling acts. See next section for details.
``` bash
python scripts/convert.py --csv_path /workspace/acts-aas/ActsExaTrkXStandalone/run/acts_nevt1_npu10_ttbar/train_all/event000000000-spacepoint.csv
mkdir -p run/standalone_nevt1_npu10_ttbar/
build/ActsExaTrkXStandalone run/acts_nevt1_npu10_ttbar/train_all/event000000000-spacepoint-converted.csv | tee run/standalone_nevt1_npu10_ttbar/log.ActsExaTrkXStandalone.txt

python scripts/convert.py --csv_path /workspace/acts-aas/ActsExaTrkXStandalone/run/acts_nevt1_npu200_ttbar/train_all/event000000000-spacepoint.csv
mkdir -p run/standalone_nevt1_npu200_ttbar
build/ActsExaTrkXStandalone run/acts_nevt1_npu200_ttbar/train_all/event000000000-spacepoint-converted.csv | tee run/standalone_nevt1_npu200_ttbar/log.ActsExaTrkXStandalone.txt
```

# Acts

## Compile Acts
``` bash
cd /workspace/acts-aas
source Scripts/setup_env.cfg

# build acts with ODD and ExaTrkX, python examples
./Scripts/make_acts_complete.sh

source Scripts/setup_acts.sh
```
## Run Acts
``` bash
cd /workspace/acts-aas/ActsExaTrkXStandalone
mkdir -p run/acts_nevt1_npu10_ttbar
python scripts/inference.py 1 /global/cfs/projectdirs/m3443/data/ACTS-aaS/models/smeared_hits/ smear --npu 10 --verbose --output run/acts_nevt1_npu10_ttbar | tee run/acts_nevt1_npu10_ttbar/log.inference.py.txt

mkdir -p run/acts_nevt1_npu200_ttbar
python scripts/inference.py 1 /global/cfs/projectdirs/m3443/data/ACTS-aaS/models/smeared_hits/ smear --npu 200 --verbose --output run/acts_nevt1_npu200_ttbar | tee run/acts_nevt1_npu200_ttbar/log.inference.py.txt

```

