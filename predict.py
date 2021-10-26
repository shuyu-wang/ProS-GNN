
import argparse
import numpy as np
from dataset import Dataset, collate_fn
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import torch.nn as nn
from torch.utils.data import DataLoader 
from models import gnn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', help= "Disables CUDA training", action='store_true', default=False)
parser.add_argument("--ngpu", help= "number of gpu", type=int, default = 1)
parser.add_argument("--ddg_fpath", help="file path of ddg",type=str,default='ddg/')
parser.add_argument("--wild_pdb", help="file path of wild_pdb",type=str,default='wild_pdb/')
parser.add_argument("--test_keys", help= "test keys", type=str, default='keys/test_keys.pkl') 
parser.add_argument("--data_fpath", help= "file path of data", type=str, default='mutation_pdb')
parser.add_argument("--models",help="test models",type=str, default='models of predict ddg ')
parser.add_argument("--batch_size", help= "batch_size", type=int, default =112)
parser.add_argument("--num_workers", help= "number of workers", type=int, default = 0)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.ngpu>0:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)

test_dataset = Dataset(test_keys, args.data_fpath, args.ddg_fpath,args.wild_pdb)
test_dataloader = DataLoader(test_dataset, args.batch_size, \
    shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = nn.MSELoss()
model = torch.load('models_name')

model.eval()
list1_test = []
list2_test = []

for i_batch, sample in enumerate(test_dataloader):
    H1,H2 , A1, A2,D1,D2, labels, key = sample
    labels = torch.Tensor(labels)
    H1,H2,A1,A2,D1,D2,labels=H1.to(device),H2.to(device),A1.to(device),A2.to(device),D1.to(device),D2.to(device),labels.to(device)
    pred =  model.test_model((H1,H2, A1,A2,D1,D2))
    loss = loss_fn(pred, labels)
    labels=labels.data.cpu().numpy()
    pred=pred.data.cpu().numpy()
    list1_test=np.append(list1_test,labels)
    list2_test=np.append(list2_test,pred)
    acc=pred/labels
rp_test = np.corrcoef(list2_test, list1_test)[0,1]
x = np.array(list1_test).reshape(-1,1)
y = np.array(list2_test).reshape(-1,1)
model = LinearRegression()
model.fit(x, y)
predict_y = model.predict(x)
predictions = {}
predictions['intercept'] = model.intercept_ 
predictions['coefficient'] = model.coef_    
predictions['predict_value'] = predict_y
print('test_corrcoef',rp_test,'rmse',np.sqrt(((y - x) ** 2).mean()))

