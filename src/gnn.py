# -*- coding: utf-8 -*-
# ! pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html | tail -n 1
# ! pip install networkx==3.1 | tail -n 1
import dgl
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.approximation.clique import maximum_independent_set as mis
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, defaultdict
from itertools import chain, islice, combinations
from time import time
from tqdm import tqdm

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv

from src import utils

# +
class GCN_dev(nn.Module):
    def __init__(self, 
                 in_feats, 
                 hidden_size, 
                 number_classes, 
                 dropout, 
                 device):
        super(GCN_dev, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)

    def forward(self, dgl_graph, inputs):
        h = self.conv1(dgl_graph, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)
        h = self.conv2(dgl_graph, h)
        h = torch.sigmoid(h)
        return h


class GNNSage_dev(nn.Module):
    def __init__(self, 
                 in_feats, 
                 hidden_size, 
                 num_classes, 
                 dropout, 
                 device, 
                 agg_type='mean', 
                 feat_drop=0):
        super(GNNSage_dev, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu, feat_drop=feat_drop)).to(device)
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type, feat_drop=feat_drop)).to(device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, dgl_graph, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(dgl_graph, h)
        h = torch.sigmoid(h)
        return h


# -

def fit_model(model, 
              dgl_graph, 
              embedding,
              loss_func,
              num_epoch=100, 
              lr=1e-3, 
              weight_decay=1e-2,
              tol=1e-5, 
              patience=1000, 
              device="cpu",
              annealing=False,
              init_reg_param=0,
              annealing_rate=1e-6,  
              check_interval=5000,
              curve_rate=2
             ):
    reg_param_state=init_reg_param
    params = chain(model.parameters(), embedding.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)    
    prev_reg_term, prev_loss, count = 1., 0, 0
    inputs=embedding.weight
    print("【START】")
    best_bit_string=model(dgl_graph, inputs)[:, 0]
    best_loss, best_cost, best_reg_term = loss_func(best_bit_string,  
                                                    reg_param_state,
                                                    curve_rate=curve_rate
                                                   )
    runtime_start = time()
    model.train()
    for epoch in range(num_epoch):
        
        probs=model(dgl_graph, inputs)[:, 0]
        loss, cost, reg_term = loss_func(probs,  
                                         reg_param_state,
                                         curve_rate=curve_rate
                                        )
        loss_ = loss.detach().item()
        reg_term_=reg_term.detach().item()
        bit_string = (probs.detach() >= 0.5)*1
        if loss < best_loss:
            best_loss=loss
            best_cost=cost
            best_reg_term=reg_term
            best_bit_string=bit_string
        if abs(reg_term_-prev_reg_term) <=tol and abs(loss_-prev_loss)<=tol:
            count += 1
        else:
            count = 0
        if count >= patience:
            print(f"Early Stopping {epoch}")
            break
        prev_reg_term = reg_term_
        prev_loss = loss_
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch%check_interval== 0:
            print(f"【TRAIN EPOCH {epoch}】LOSS {loss:.3f} COST {cost:.3f} REG {reg_term:.3f} PARAM {reg_param_state:.3f}")
        if annealing:
            reg_param_state += annealing_rate
    runtime = time() - runtime_start
    return model, bit_string, cost, reg_term, runtime
