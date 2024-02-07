# Perf_analyzer
## Prepare the perf_analyzer input
``` bash
find /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs -name "event*-spacepoint.csv" -type f | xargs -I {} python /workspace/ActsExaTrkXStandalone/scripts/convert.py --csv_path {}

python scripts/gen_json.py --csv_path /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs

perf_analyzer -m ActsExaTrkX --percentile=95 -i grpc --input-data /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs/event000000000-spacepoint-converted.json
/global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs/ttbarN100PU200_SPs.json
```

## Benchmark on Interactive nodes
Since it usually takes a long time before a slurm job starts, running benchark testing on interactive nodes is recommended.
After getting an interactive node, run shifter and then execute the script, e.g. `.script/evaluate_triton.sh 1 1`.