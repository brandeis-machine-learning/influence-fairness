# Achieving Fairness at No Utility Cost via Data Reweighing with Influence

Code for ICML 2022 paper: Achieving Fairness at No Utility Cost via Data Reweighing with Influence.  
https://proceedings.mlr.press/v162/li22p/li22p.pdf

![Drag Racing](./fig/icml_2022_poster.png)

Please consider citing our paper if you find our research helpful :)

### Reference

```
@InProceedings{pmlr-v162-li22p,
  title = 	 {Achieving Fairness at No Utility Cost via Data Reweighing with Influence},
  author =       {Li, Peizhao and Liu, Hongfu},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {12917--12930},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/li22p/li22p.pdf},
  url = 	 {https://proceedings.mlr.press/v162/li22p.html},
  abstract = 	 {With the fast development of algorithmic governance, fairness has become a compulsory property for machine learning models to suppress unintentional discrimination. In this paper, we focus on the pre-processing aspect for achieving fairness, and propose a data reweighing approach that only adjusts the weight for samples in the training phase. Different from most previous reweighing methods which usually assign a uniform weight for each (sub)group, we granularly model the influence of each training sample with regard to fairness-related quantity and predictive utility, and compute individual weights based on influence under the constraints from both fairness and utility. Experimental results reveal that previous methods achieve fairness at a non-negligible cost of utility, while as a significant advantage, our approach can empirically release the tradeoff and obtain cost-free fairness for equal opportunity. We demonstrate the cost-free fairness through vanilla classifiers and standard training processes, compared to baseline methods on multiple real-world tabular datasets. Code available at https://github.com/brandeis-machine-learning/influence-fairness.}
}
```

## Setup

### Dataset

Download data to `./data` and build the folder structure as

```
./data
    └ adult
    │    │ adult.data
    │    │ adult.test
    │    └ meta.json
    └ compas
    │    │ compas-scores-two-years.csv
    │    │ idx.json
    │    └ meta.json
    └ german
         │ german.data
         │ idx.json
         └ meta.json
```

Adult: https://archive.ics.uci.edu/ml/datasets/adult  
Compas: https://github.com/propublica/compas-analysis/  
German: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

### Gurobi

We use [Gurobi](https://www.gurobi.com/) to solve linear programs.

Install Gurobi for
Python: https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-  
We register an Individual Academic License to use the full features of Gurobi. Please
follow https://www.gurobi.com/academia/academic-program-and-licenses/ to obtain a `gurobi.lic` if you are eligible.

## Experiment

Run logistic regression on the Adult dataset with Equal Opportunity.

```
python main.py --dataset adult --metric eop --beta 0.5 --gamma 0.2
```

You will have the following printing information:

```
Loading Adult dataset..
Dataset statistic - #total: 45222; #train: 22622; #val.: 7540; #test: 15060; #dim.: 102

Maximum fairness promotion: -0.71638; Maximum utility promotion: -241.51456
Set parameter Username
Academic license - for non-commercial use only - expires 2022-12-01
Warning for adding constraints: zero or small (< 1e-13) coefficients, ignored
Gurobi Optimizer version 9.5.0 build v9.5.0rc5 (linux64)
Thread count: 10 physical cores, 20 logical processors, using up to 20 threads
Optimize a model with 2 rows, 22622 columns and 44984 nonzeros
Model fingerprint: 0x87f80f7c
Coefficient statistics:
  Matrix range     [1e-13, 2e+00]
  Objective range  [1e+00, 1e+00]
  Bounds range     [1e+00, 1e+00]
  RHS range        [9e-02, 5e+01]

Concurrent LP optimizer: primal simplex, dual simplex, and barrier
Showing barrier log only...

Presolve removed 0 rows and 8933 columns
Presolve time: 0.02s
Presolved: 2 rows, 13689 columns, 27378 nonzeros

Ordering time: 0.00s

Barrier statistics:
 AA' NZ     : 1.000e+00
 Factor NZ  : 3.000e+00 (roughly 5 MB of memory)
 Factor Ops : 5.000e+00 (less than 1 second per iteration)
 Threads    : 1

                  Objective                Residual
Iter       Primal          Dual         Primal    Dual     Compl     Time
   0   6.67687894e+09 -2.82255343e+11  1.19e+00 6.04e+05  7.28e+08     0s
   1   5.71993064e+07 -2.49057677e+11  1.12e-02 8.38e+06  1.35e+07     0s
   2   1.58465157e+05 -4.06195963e+10  3.36e-05 1.07e-06  1.49e+06     0s
   3   7.21509681e+03 -6.37481572e+06  0.00e+00 5.96e-07  2.33e+02     0s
   4   7.05936180e+03 -1.56293507e+05  0.00e+00 3.73e-07  5.97e+00     0s
   5   5.47000038e+03 -2.77631209e+04  0.00e+00 6.61e-08  1.21e+00     0s
   6   4.54852224e+03 -1.00437285e+04  0.00e+00 5.96e-08  5.33e-01     0s

Barrier performed 6 iterations in 0.03 seconds (0.03 work units)
Barrier solve interrupted - model solved by another algorithm


Solved with dual simplex
Solved in 7 iterations and 0.03 seconds (0.02 work units)
Optimal objective  2.497175044e+02
Total removal: 249.71750; Ratio: 1.104%

Fairness loss: 0.17708 -> 0.08392; Utility loss: 3050.91919 -> 3027.89156
------------------------------ Results on val
Grp. 0 - #instance: 2446; #pos. pred: 357; Acc.: 0.899428
Grp. 1 - #instance: 5094; #pos. pred: 1684; Acc.: 0.797801
Overall acc.: 0.830769; Demographic parity: 0.184632; Equal opportunity: 0.007115
------------------------------ Results on test
Grp. 0 - #instance: 4913; #pos. pred: 750; Acc.: 0.889477
Grp. 1 - #instance: 10147; #pos. pred: 3229; Acc.: 0.795210
Overall acc.: 0.825963; Demographic parity: 0.165566; Equal opportunity: -0.002712
Total time: 2.64818s
```
