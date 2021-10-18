import utils
import time
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
if __name__ == "__main__":

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print (s)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ddg_fpath", help="file path of ddg",type=str,default='ddg/')
    parser.add_argument("--wild_pdb", help="file path of wild_pdb",type=str,default='wild_pdb/')
    parser.add_argument("--mutation_pdb", help= "file path of data", type=str, default='mutation_pdb')    
    parser.add_argument("--train_keys", help= "train keys", type=str, default='keys/train_keys.pkl')
    parser.add_argument("--test_keys", help= "test keys", type=str, default='keys/test_keys.pkl') 
    parser.add_argument('--seed', help= "Random seed", type=int, default=42)  
    parser.add_argument("--num_workers", help= "number of workers", type=int, default = 0) 
    parser.add_argument("--ngpu", help= "number of gpu", type=int, default = 0)
    parser.add_argument('--epochs', help= "Number of epochs to train", type=int, default = 400)
    parser.add_argument("--lr", help= "learning rate", type=float, default = 0.0001)
    parser.add_argument("--batch_size", help= "batch_size", type=int, default = 16) 
    parser.add_argument("--dropout_rate", help= "dropout_rate", type=float, default = 0.5)
    parser.add_argument('--weight_decay', help= "Weight decay (L2 loss on parameters)", type=float, default=5e-5)
    parser.add_argument("--n_graph_layer", help= "number of GNN layer", type=int, default = 4)  
    parser.add_argument("--d_graph_layer", help= "dimension of GNN layer", type=int, default =1120)   
    parser.add_argument("--n_FC_layer", help= "number of FC layer", type=int, default = 4)
    parser.add_argument("--d_FC_layer", help= "dimension of FC layer", type=int, default =1024)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    if args.ngpu>0:
            os.environ['CUDA_VISIBLE_DEVICES']='0'
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    ddg_data_fpath=args.ddg_fpath
    wild_pdb_fpath=args.wild_pdb
    mutation_pdb_fpath= args.mutation_pdb

    model = gnn(args)
    with open (args.train_keys, 'rb') as fp:
            train_keys = pickle.load(fp)
    with open (args.test_keys, 'rb') as fp:
            test_keys = pickle.load(fp)
    train_dataset = Dataset(train_keys, args.mutation_pdb, args.ddg_fpath,args.wild_pdb)
    test_dataset  = Dataset(test_keys, args.mutation_pdb, args.ddg_fpath,args.wild_pdb) 

    train_dataloader = DataLoader(train_dataset, args.batch_size, \
        shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    for epochs in range(num_epochs):

        list1_test = []
        list2_test = []
        list1_train = []
        list2_train = []
        test_losses = []    
        train_losses = []
        st = time.time()
        model.train()    
        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            H1, H2, A1, A2, P1, P2, labels, key = sample
            labels = torch.Tensor(labels)
            H1,H2 , A1, A2,P1,P2, labels=H1.to(device),H2.to(device),A1.to(device),A2.to(device),P1.to(device),P2.to(device),labels.to(device)
            pred = model.train_model(H1, H2, A1, A2, P1, P2)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data.cpu().numpy())
            pred=pred.data.cpu().numpy()
            labels=labels.data.cpu().numpy()
            list1_train=np.append(list1_train,labels)
            list2_train=np.append(list2_train,pred)

        model.eval()
        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()
            H1,H2 , A1, A2,P1,P2, labels, key = sample
            labels = torch.Tensor(labels)
            H1,H2 , A1, A2,P1,P2, labels=H1.to(device),H2.to(device),A1.to(device),A2.to(device),P1.to(device),P2.to(device),labels.to(device)
            pred = model.train_model(H1, H2, A1, A2, P1, P2)
            loss = loss_fn(pred, labels)
            test_losses.append(loss.data.cpu().numpy())
            labels=labels.data.cpu().numpy()
            pred=pred.data.cpu().numpy()
            list1_test=np.append(list1_test,labels)
            list2_test=np.append(list2_test,pred)
            acc=pred/labels        
        rp_train = np.corrcoef(list2_train, list1_train)[0,1]
        rp_test = np.corrcoef(list2_test, list1_test)[0,1]
        test_losses = np.mean(np.array(test_losses))
        train_losses = np.mean(np.array(train_losses))
        x = np.array(list1_test).reshape(-1,1)
        y = np.array(list2_test).reshape(-1,1)
        rmse= np.sqrt(((y - x) ** 2).mean())
        end = time.time()
        print('epochs  train_losses   test_losses     pcc_train        pcc_test        rmse           end-st')
        print ("%s  \t%.3f   \t%.3f   \t%.3f   \t%.3f   \t%.3f    \t%.3f" \
        %(epochs, train_losses,     test_losses,     rp_train,     rp_test,     rmse,   end-st))
















