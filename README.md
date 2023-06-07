# MIDAM
The official implementation of 'Provable Multi-instance Deep AUC Maximization with Stochastic Pooling', ICML2023

Here are some dependencies (with the version for the experiments results reported in the paper): torch==1.9.0, numpy==1.17.4, CUDA version:12.0 on NVIDIA Quadro RTX 6000 card.

This is the code that runs MIDAM with stochastic softmax or attention pooling on benchmark datasets, such as MUSK1&2, Fox, Tiger, Elephant, Breast Cancer, etc. 

## Examples: 

### Run MIDAM with stochastic attention pooling on Fox data

```
CUDA_VISIBLE_DEVICES=0 python3 experiment.py --dataset=Fox  --loss=MIDAM-att  --momentum=0.1  --seed=123 --lr=1e-2 --bag_batch_size=4
```

### Run MIDAM with stochastic softmax pooling on Fox data

```
CUDA_VISIBLE_DEVICES=0 python3 experiment.py --dataset=Fox  --loss=MIDAM-smx  --momentum=0.1  --seed=123 --lr=1e-2 --bag_batch_size=4
```


## Others:
Please make sure you have the data on the data folder (some smaller datasets are already included). Please refer to the experiment.py and datasets.py for how to load the data. 


## Citation:
```
@inproceedings{zhu2023provable,
	title={Provable Multi-instance Deep AUC Maximization with Stochastic Pooling},
	author={Dixian Zhu and Bokun Wang and Zhi Chen and Yaxing Wang and Milan Sonka and Xiaodong Wu and Tianbao Yang},
	booktitle={Proceedings of the 40th International Conference on Machine Learning},
	year={2023}
	}  
```

