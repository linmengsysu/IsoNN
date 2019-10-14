import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math 
import numpy as np
from itertools import permutations

from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

import random

class Isomorphic_Feature_Extraction(nn.Module):
    """docstring for ClassName"""
    def __init__(self, k, c):
        super(Isomorphic_Feature_Extraction, self).__init__()
        self.k = k
        self.c = c
        self.fac_k = math.factorial(k)
        self.P = self.get_all_P(self.k)

        self.kernel1 = nn.Parameter(torch.randn(c, k, k))
        self.kernel2 = nn.Parameter(torch.randn(c, k, k))

        self.maxpoolk = nn.MaxPool2d((1, self.fac_k), stride=(1, self.fac_k))

  

    def forward(self, x):
        layer1 = self.FastIsoLayerWithP(x, self.kernel1)
        layer1 = self.IsoLayer(x, self.kernel1)
        B, _, c= layer1.size()
        out = F.softmax(-1*layer1, dim=-2).view(B, -1) 
        return out

    def IsoLayer(self, x, kernel):
        print('IsoLayer')
        x = self.get_all_subgraphs(x, self.k)
        B, n_subgraph, k, k = x.size()
        x = x.view(-1, 1, self.k, self.k)
        H = int(np.sqrt(n_subgraph))
        tmp = torch.matmul(torch.matmul(self.P, kernel.view(self.c, 1, self.k, self.k)), torch.transpose(self.P, 2, 1)).view(-1, self.k, self.k) - x #[B*n_subgraph, 1, k, k] - [c*k!, k, k]
        raw_features = -1 * torch.norm(tmp, p='fro', dim=(-2,-1)) ** 2
        raw_features = raw_features.view(B, n_subgraph, self.c, self.fac_k)
        
        feature_P = self.maxpoolk(raw_features).view(B, -1, self.c)
        feature_P = (-1) * feature_P

        return feature_P


    def FastIsoLayerWithP(self, x, kernel):
        print('fast IsoLayer')
        x = self.get_all_subgraphs(x, self.k)
        
        B, n_subgraph, k, k = x.size()
        x = x.view(-1, self.k, self.k)
        P = self.compute_p(x, kernel)# P [B*n_subgraph, c, k, k] 
        x = x.view(-1, 1, self.k, self.k)
        tmp = x - torch.matmul(torch.matmul(P, kernel), torch.transpose(P, -2, -1))#[B*n_subgraph, 1, k, k] - [B*n_subgraph, c, k, k] 

        features = torch.norm(tmp, p='fro', dim=(-2,-1)) ** 2
        features = features.view(B, n_subgraph, self.c)
        return features
  
    def compute_p(self, subgraphs, kernel): 
        c, k, k = kernel.size()
        N, k, k = subgraphs.size() # N = B * n_subgraph
        VGs, UGs = LA.eig(subgraphs.detach().numpy()) 
        VHs, UHs = LA.eig(kernel.detach().numpy())

        bar_UGs = np.absolute(UGs).reshape(-1, 1, k, k)
        bar_UHs = np.absolute(UHs)

        P = np.matmul(bar_UGs, np.transpose(bar_UHs,(0,2,1)))
        P_star = torch.from_numpy(np.array(P)).requires_grad_(False)
        P_star = P_star.type(torch.FloatTensor)
        return P_star
        

    # get all possible P (slow algo)
    def get_all_P(self, k):
        n_P = np.math.factorial(k)
        P_collection = np.zeros([n_P, k, k])
        perms = permutations(range(k), k)

        count = 0
        for p in perms:
            for i in range(len(p)):
                P_collection[count, i, p[i]] = 1
            count += 1
        Ps = torch.from_numpy(np.array(P_collection)).requires_grad_(False)
        Ps = Ps.type(torch.FloatTensor)

        return Ps

    def get_all_subgraphs(self, X, k):
        X = X.detach().squeeze()
        (batch_size, n_H_prev, n_W_prev) = X.size()

        n_H = n_H_prev - k + 1
        n_W = n_W_prev - k + 1
        subgraphs = []
        for h in range(n_H):
            for w in range(n_W):
                x_slice = X[:, h:h+k, w:w+k]
                subgraphs.append(x_slice)
        S = torch.stack(subgraphs, dim=1)
        return S



class Classification_Component(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, nclass):
        super(Classification_Component, self).__init__()
        self.input_size = input_size


        self.fc1 = nn.Linear(input_size, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, nclass)

    def forward(self, features):
        h1 = F.relu(self.fc1(features))
        h2 = F.relu(self.fc2(h1))
        pred = F.log_softmax(self.fc3(h2), dim=1)
        return pred







