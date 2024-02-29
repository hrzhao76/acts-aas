# Perf_analyzer
## Prepare the perf_analyzer input
**No need to do this for the spacepoints(csv format) in the shared folder.**

``` bash
find /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs -name "event*-spacepoint.csv" -type f | xargs -I {} python /workspace/ActsExaTrkXStandalone/scripts/convert.py --csv_path {}

python scripts/gen_json.py --csv_path /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs

perf_analyzer -m ActsExaTrkX --percentile=95 -i grpc --input-data /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs/event000000000-spacepoint-converted.json
/global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs/ttbarN100PU200_SPs.json

shifter --module=gpu --volume=${PWD}:/workspace/ --workdir=/workspace/ --image=nvcr.io/nvidia/tritonserver:22.02-py3-sdk /bin/bash
```

## Benchmark on Interactive nodes
Since it usually takes a long time before a slurm job starts, running benchark testing on interactive nodes is recommended.
This will give you exactly one GPU
``` bash
srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l
```

After getting an interactive node, run shifter and then execute the script, e.g. `.scripts/evaluate_triton.sh 1 1`.
One can also run `.scripts/run_evaluate.sh` to test different number of instances on 1 GPU. But this script sometimes donesn't successfully generate an output csv. In future a python based script might be needed here.
Note based on the 100 events to be used as benchmark(it ensures the variaties of events), measurement-interval should be long enough to loop through all events to get statble results.

### Plotting
Refer to [plot_1GPU.ipynb](./plotting/plot_1GPU.ipynb).
