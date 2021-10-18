import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
import torch.nn as nn
class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)

    def forward(self, x, adj):
        h = self.W(x)   
        h_prime = F.leaky_relu(torch.einsum('aij,ajk->aik',(adj, h)))
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval



