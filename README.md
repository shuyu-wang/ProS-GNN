# ProS-GNN

This repo contains the code for our paper " ProS-GNN: Predicting effects of mutations on protein stability using graph neural networks"

by Shuyu Wang*, Hongzhou Tang

Here we pioneered a deep graph neural network based method for predicting protein stability change upon mutation. After mutant part data extraction, the model encoded the molecular structure-property relationships using message passing and incorporated raw atom coordinates to enable spatial insights into the molecular systems. We trained the model using the S2648 and S3412 datasets, and tested on the Ssym and Myoglobin datasets. Compared to existing methods, our proposed method showed competitive high performance in data generalization and bias suppression with ultra-low time consumption.
(/fig1(A).png)

#Dependency
-Python 3.7
-Pytorch
-numpy
-RDKit
-sklearn
-CUDA

#Usage
S3214 dataset for training can be found at: https://github.com/gersteinlab/ThermoNet
To train the model:
<python train.py>
To try with an example:
