import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate

N_atom_features=30
class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate
        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)])        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1]+3, d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.embede = nn.Linear(N_atom_features, d_graph_layer, bias = False)

    def embede_graph(self, data):
        h_m,h_w, adj_m,adj_w,p_m,p_w = data
        h_m = self.embede(h_m) 
        h_w = self.embede(h_w) 
        n1=h_m.shape[1]
        n2=h_w.shape[1]
        bn1 = nn.BatchNorm1d(n1)
        bn2 = nn.BatchNorm1d(n2)
        bn1.cuda()
        bn2.cuda()           
        h_m_sc=h_m
        h_w_sc=h_w
        for k in range(len(self.gconv1)):
            h_m = self.gconv1[k](h_m, adj_m)
            h_m=bn1(h_m)
            h_w = self.gconv1[k](h_w, adj_w)
            h_w=bn2(h_w) 
        h_m=h_m+h_m_sc
        h_w=h_w+h_w_sc
        h_m=torch.cat((h_m,p_m),2)
        h_w=torch.cat((h_w,p_w),2)
        h_m = h_m.sum(1)
        h_w = h_w.sum(1)
        h = h_m-h_w
        return h

    def fully_connected(self, h):
        n=h.shape[0]
        fc_bn = nn.BatchNorm1d(n)
        fc_bn.cuda()
        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                h = self.FC[k](h)
                h=h.unsqueeze(0)
                h=fc_bn(h)
                h=h.squeeze(0)
                h = F.dropout(h, p=self.dropout_rate, training=self.training)
                h = F.leaky_relu(h)
            else:
                h = self.FC[k](h)
        return h

    def train_model(self, data):
        h = self.embede_graph(data)
        h = self.fully_connected(h)
        h = h.view(-1)
        return h
    
    def test_model(self,data):
        h = self.embede_graph(data)
        h = self.fully_connected(h)
        h = h.view(-1)
        return h
