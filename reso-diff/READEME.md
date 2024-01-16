# Resolve Difference(acts_demo and exatrkx-acts-demonstrator)

## Official Acts v29.2.0

``` bash
git clone https://github.com/acts-project/acts.git
cd acts
git submodule init
git submodule update
```

## Acts-aas 
```
cd /workspace/acts-aas/ 
git checkout dev/reso-diff 
cd reso-diff

. make_acts.sh  
```

### Exatrkx-Acts-demonstrator
``` bash
mkdir -p run/test_npu10_truth run/test_npu10_smear
python scripts/inference.py 1 $DEMO_PATH/models/true_hits/ truth --output run/test_npu10_truth --verbose | tee run/test_npu10_truth/log.txt 

python scripts/inference.py 1 $DEMO_PATH/models/smeared_hits/ smear --output run/test_npu10_smear --verbose | tee run/test_npu10_smear/log.txt

python scripts/inference.py 1 $DEMO_PATH/models/smeared_hits/ smear --output run/test_npu0_smear --verbose | tee run/test_npu0_smear/log.txt
python scripts/convert.py 

```

### Acts_demo 

``` bash 
inference-gpu -e acts-smear -m /workspace/exatrkx-service/acts_demostrator/models/smeared_hits/ -d /workspace/exatrkx-service/exatrkx_pipeline/datanmodels/in_e1000.csv 

inference-gpu -e acts-smear -m /workspace/exatrkx-service/acts_demostrator/models/smeared_hits/ -d /workspace/acts-aas/reso-diff/run/test_npu10_smear/train_all/event000000000-spacepoint-converted.csv 

inference-gpu -e acts-smear -m /workspace/exatrkx-service/acts_demostrator/models/smeared_hits/ -d /workspace/acts-aas/reso-diff/run/test_npu0_smear/train_all/event000000000-spacepoint-converted.csv 

```