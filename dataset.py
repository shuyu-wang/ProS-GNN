from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle

def get_atom_feature(m):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(utils.atom_feature(m, i, None, None))
    H = np.array(H)
    return H+0        

class Dataset(Dataset):

    def __init__(self, keys, data_dir,ddg_dir,wild_dir):
        self.keys = keys
        self.data_dir = data_dir
        self.ddg_dir = ddg_dir
        self.wild_dir = wild_dir
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]

        mol_w=Chem.MolFromPDBFile('./'+self.wild_dir+'/'+key+'_wild.pdb')
        mol_m=Chem.MolFromPDBFile('./'+self.data_dir+'/'+key+'_mutation.pdb')
        with open('./'+ self.ddg_dir +'/'+key, 'rb') as f:
            labels = pickle.load(f)

        #mutation type information
        n_m = mol_m.GetNumAtoms()
        c_m = mol_m.GetConformers()[0]
        p_m = np.array(c_m.GetPositions())  
        adj_m = GetAdjacencyMatrix(mol_m)+np.eye(n_m)
        H_m = get_atom_feature(mol_m)

        #wild type information
        n_w = mol_w.GetNumAtoms()
        c_w = mol_w.GetConformers()[0]
        P_w = np.array(c_w.GetPositions())  
        adj_w = GetAdjacencyMatrix(mol_w)+np.eye(n_w)
        H_w = get_atom_feature(mol_w)
        labels=labels
        sample = {'H_m': H_m,\
                  'H_w': H_w,\
                  'A1': adj_m, \
                  'A2': adj_w, \
                  'P_m': p_m,\
                  'P_w': P_w,\
                  'labels': labels, \
                  'key': key, \
                  }
        return sample



def collate_fn(batch):
    max_natoms_m = max([len(item['H_m']) for item in batch if item is not None])
    max_natoms_w = max([len(item['H_w']) for item in batch if item is not None])
    H_m = np.zeros((len(batch), max_natoms_m, 30))
    H_w = np.zeros((len(batch), max_natoms_w, 30))
    A1 = np.zeros((len(batch), max_natoms_m, max_natoms_m))
    A2 = np.zeros((len(batch), max_natoms_w, max_natoms_w))
    p_m = np.zeros((len(batch), max_natoms_m, 3))
    P_w = np.zeros((len(batch), max_natoms_w, 3))

    keys = [] 
    labels=[]   
    for i in range(len(batch)):
        natom1 = len(batch[i]['H_m'])
        natom2 = len(batch[i]['H_w'])        
        H_m[i,:natom1] = batch[i]['H_m']
        H_w[i,:natom2] = batch[i]['H_w']
        A1[i,:natom1,:natom1] = batch[i]['A1']
        A2[i,:natom2,:natom2] = batch[i]['A2']
        p_m[i,:natom1,:natom1] = batch[i]['p_m']
        P_w[i,:natom2,:natom2] = batch[i]['P_w']
        keys.append(batch[i]['key'])
        labels.append(batch[i]['labels'])
    H_m = torch.from_numpy(H_m).float()
    H_w = torch.from_numpy(H_w).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    P_m = torch.from_numpy(p_m).float()
    P_w = torch.from_numpy(P_w).float()

    return H_m, H_w, A1, A2,P_m,P_w, labels, keys