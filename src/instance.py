# -*- coding: utf-8 -*-

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

from dgl.nn import GATConv
from dgl.nn.pytorch import GraphConv

from src import utils

def gen_q_dict_mis_sym(nx_G, penalty=2):
    Q_dic = defaultdict(int)
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty
        Q_dic[(v, u)] = penalty
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1
    return Q_dic


def gen_q_dict_max_cut_sym(nx_G):
    Q_dic = defaultdict(int)
    for u, v in nx_G.edges:
        Q_dic[(u,u)]+= -1
        Q_dic[(v,v)]+= -1
        Q_dic[(u,v)]+= 2
        Q_dic[(v,u)]+= 2
    return Q_dic
