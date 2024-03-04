# Kubernetes Notes
``` bash
kubectl create -f acts-triton-pod.yaml
kubectl create -f acts-triton-dep.yaml 
kubectl create -f acts-triton-svc.yaml
kubectl create -f acts-triton-ingress.yaml
```

## Useful Commands

``` bash
kubectl get pod -o wide acts-triton-build
kubectl describe pod acts-triton-build
kubectl get svc
kubectl get deployments
kubectl logs <pod-name> #--previous


kubectl exec -it acts-triton-build -- /bin/bash

root@acts-triton-build:/workspace# source /workspace/acts-aas/Scripts/setup_env.cfg
root@acts-triton-build:/workspace# cd $INSTALLDIR
root@acts-triton-build:/workspace# tritonserver --model-repository=$INSTALLDIR/model_repo/ --log-verbose=4
```
### Test Connection [Only works with v1 ingress]
``` bash
curl -v https://acts-triton.nrp-nautilus.io/v2/health/ready
# I0304 08:58:37.022385 103 http_server.cc:2976] HTTP request: 0 /v2/health/ready
# I0304 08:58:37.022446 103 model_repository_manager.cc:522] ModelStates()
```
Output 200 OK
```
> GET /v2/health/ready HTTP/2
> Host: acts-triton.nrp-nautilus.io
> User-Agent: curl/8.4.0
> Accept: */*
>
< HTTP/2 200
< content-length: 0
< content-type: text/plain
< strict-transport-security: max-age=15768000
<
* Connection #0 to host acts-triton.nrp-nautilus.io left intact
```

### Test Inference
``` bash
cd Clients
# pip instlll tritonclient[all]
python ActsExaTrkXTritionClient.py -u acts-triton.nrp-nautilus.io


# Only works with ingress-v1
perf_analyzer -m ActsExaTrkX --input-data /global/cfs/projectdirs/m3443/data/ACTS-aaS/ttbarN100PU200_SPs/event000000000-spacepoint-converted.json -u acts-triton.nrp-nautilus.io

# grpc protocol
perf_analyzer -m ActsExaTrkX -i grpc --ssl-grpc-use-ssl --percentile=95 --input-data /workspace/acts-aas/Clients/event000000000-spacepoint-converted.json -u acts-triton.nrp-nautilus.io:443


perf_analyzer -m ActsExaTrkX --percentile=95 \
-i grpc --ssl-grpc-use-ssl --input-data ../ttbarN100PU200_SPs.json -u acts-triton.nrp-nautilusio:443 \
--measurement-interval 100000 --sync --concurrency 1:10:1 -b 1 --collect-metrics -f result.csv --verbose-csv

```

## Namespace Monitor

### Grafana  
Go to the [Grafana](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods?orgId=1&refresh=10s&var-datasource=default&var-cluster=&var-namespace=extraks-aas) page.  

## References
[Scaling and exposing](https://docs.nationalresearchplatform.org/userdocs/tutorial/basic2/)   
[GPU Pods](https://docs.nationalresearchplatform.org/userdocs/running/gpu-pods/)