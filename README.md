## Introduction
Accurate prediction of the energy characteristics of molecules is important in various fields including quantum chemistry, catalysis, materials science and drug design. Traditional methods based on ab-initio calculations can be very time and computationally expensive, especially for large molecular systems.

In this project, I use [MatErials Graph Network (MEGNet)](https://github.com/materialsvirtuallab/megnet), one of the most popular mainstream graph neural networks for predicting the properties of molecules from their graph representation. The main feature of [MEGNet](https://github.com/materialsvirtuallab/megnet) in comparison to other models is its lightweight nature and its advanced use of triple-type features for each molecule, which have been pioneering in subsequent research and development, e.g., [M3GNet](https://www.nature.com/articles/s43588-022-00349-3).

The task was to create a model whose MAE would not exceed 1 *meV*, i.e., the DFT error for calculating the energy of molecules. To do this, hypotheses for molecule interactions were tested.

### Solution
Dataset analysis and baseline creation - [analisys](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/analisys.ipynb)

Model training and validation - [train](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/training.ipynb)

Training on dataset with added samples from qm9 - [train_qm9](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/train_with_qm9.ipynb)

In the folder [other models](https://github.com/Gruz2520/predict_energy_of_mols/tree/main/other_models) we tested other models for further comparative analysis of the results.

### Results
|Model|MAE|
|-----|---|
|MEGNet|0.0017 meV\atom|
|schenet|0.0042 meV\atom|
|dimenet++|0.0066 meV\atom|
|comenet|0.0031 meV\atom|

The computational results were produced on the [HSE cluster cHARISMa](https://hpc.hse.ru/en/hardware/hpc-cluster/). 

[work rep](https://github.com/Gruz2520/megnet_tests)

### Requirements

Because the main rep is no longer supported and the core libraries have jumped ahead, we had to rewrite some of the source code to make it run normally.

- Python 3.7.9

- local_env.py file for pymatgen

    path for change: ../pymatgen/analysis/local_env.py

- callbacks.py for megnet

    path for change: ../megnet/callbacks.py

```bash
pip install -r req.txt
```
