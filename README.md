
# Interpreting Unfairness in Graph Neural Networks via Training Node Attribution

Open-source code for "Interpreting Unfairness in Graph Neural Networks via Training Node Attribution".

## Citation

If you find it useful, please cite our paper. Thank you!

```
@inproceedings{dong2023bind,
  title={Interpreting Unfairness in Graph Neural Networks via Training Node Attribution},
  author={Dong, Yushun and Wang, Song and Ma, Jing and Liu, Ninghao and Li, Jundong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```




## 1. Requirements & Environment

To install requirements:

```setup
pip install -r requirements.txt
```

Experiments are carried out on a Titan RTX with Cuda 10.1. 

Library details can be found in requirements.txt.

Notice: Cuda is enabled for default training settings.



## 2. Introduction

This open-source code mainly includes five python files, namely 1_training.py, 2_influence_computation_and_save.py, 3_removing_and_testing.py, debiasing_gnns.py, and plots_effectiveness.py. We provide a basic introduction below to help gain an understanding of how they work.

#### (1) 1_training.py

This is the file to train the GNN model to be interpreted. After training, the optimized GNN will be saved for interpretation.

#### (2) 2_influence_computation_and_save.py

This is the file to estimate the node influence on bias following Eq.(5), (6), (7), and (8) in the paper.

Notice: 

(i) The "helpfulness" mentioned in the code is the node-level helpfulness to unfairness.

(ii) The hyper-parameters in approximator.py requires parameter-tuning. Example configurations for function s_test_graph_cost are provided here:
for Income and Recidivism, set the scale in function s_test_graph_cost as 25; for Pokec1, set the scale in function s_test_graph_cost as 100;
for Pokec2, set the scale in function s_test_graph_cost as 60.

(iii) The PDD is initialized with the function defined in Section *Debiasing via Harmful Nodes Deletion* by default.




#### (3) 3_removing_and_testing.py

This is the file to (1) follow the strategy introduced in Section *Effectiveness of Node Influence Estimation* to select node sets to delete; (2) obtain the actual training results when each one of the node sets is deleted from the graph; (3) combine the actual results and estimated results in a variable named final_sets.
Note that in (2), we follow a widely used training strategy to continue training based on the optimized model instead of training from scratch. This helps avoid the correlation between estimated and actual values from being swamped by randomness.
To know more about this evaluation strategy, see https://github.com/kohpangwei/influence-release/issues/16#issuecomment-548225359



#### (4) debiasing_gnns.py

This is the file to evaluate to what extent deleting those training nodes with large contributions to the exhibited bias can help debiasing the GNN.
Notice that we have already done all the training runs after deleting each obtained set of training nodes in 3_removing_and_testing.py. Thus in this script, we directly read the performances from the saved variable final_sets.


#### (5) plots_effectiveness.py

This is the file to plot and evaluate the correlation between estimated values and actual ones.


## 3. Estimation & Evaluation

Note that different seeding settings might lead to discrepancies. In the sections below, we provide an exemplary run under seed 10 with the GCN model on the Income dataset.


### 3.1 GNN Training


To train a vanilla GCN, run

```
python 1_train.py
```

and we present the sample log based on a random seed 10 as follows.

```
Time: 8.958610534667969 s
*****************  Cost  ********************
SP cost:
0.7684668737613751
EO cost:
0.9045681932082816
**********************************************
Test set results: loss= 0.5638 accuracy= 0.7070
Statistical Parity:  0.3125430281432843
Equality:  0.3597688250476939
```


### 3.2 Node Influence Estimation

To estimate the training node influence on the model bias, run

```
python 2_influence_computation_and_save.py 
```

and we present the sample log based on a random seed 10 as follows.

```
Pre-processing data...
Reconstructing the adj of income dataset...
100%|██████████| 1000/1000 [00:00<00:00, 1808.32it/s]
Pre-processing completed.
100%|██████████| 1000/1000 [05:46<00:00,  2.89it/s]
Average time per training node: 0.3576169583797455 s
```

### 3.3 Collect Actual PDD values

To collect the estimated & actual PDD values, first run:

```
python 3_removing_and_tesing.py --helpfulness_collection 0
```

Then run:

```
python 3_removing_and_tesing.py --helpfulness_collection 1
```


We present the two sample logs based on a random seed 10 as follows.

```
0
Loading income dataset from ../data/income/
Reconstructing the adj of income dataset...
Finding neighbors ... 
100%|██████████| 1000/1000 [00:00<00:00, 1716.26it/s]
At most effective number:
327
Loading income dataset from ../data/income/
100%|██████████| 99/99 [00:36<00:00,  2.73it/s]
```

```
1
Loading income dataset from ../data/income/
Reconstructing the adj of income dataset...
Finding neighbors ... 
100%|██████████| 1000/1000 [00:00<00:00, 1659.89it/s]
At most effective number:
515
Loading income dataset from ../data/income/
100%|██████████| 155/155 [00:54<00:00,  2.86it/s]
```

### 3.4 GNN Debiasing & Correlation Results

To see debiasing results, run:

```
python debiasing_gnns.py
```

We present the sample log based on a random seed 10 as follows.

```
BIND 1%:
Acc: 0.755261737722612
Statistical Parity: 0.1727779948876637
Equal Opportunity: 0.22710133542812255
BIND 10%:
Acc: 0.7193739881273611
Statistical Parity: 0.14003243931033985
Equal Opportunity: 0.15743463135450564
```

To see Pearson correlation evaluations, run:

```
python plots_effectiveness.py
```
