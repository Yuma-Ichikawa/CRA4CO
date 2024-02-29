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


def qubo_dict_to_torch(nx_G, Q, torch_dtype=torch.float32, device="cpu"):
    n_nodes = len(nx_G.nodes)
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val
    return Q_mat.type(torch_dtype).to(device)

def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


def is_symmetric(matrix):
    if matrix.size(0) != matrix.size(1):
        return False
    diff = torch.abs(matrix - matrix.t())
    return torch.max(diff).item() < 1e-6


def postprocess_gnn_mis(best_bit_string, nx_graph):
    bitstring_list = list(best_bit_string)
    size_mis = sum(bitstring_list)

    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations

def postprocess_gnn_max_cut(best_bit_string, nx_graph):
    bitstring_list = list(best_bit_string)
    S0 = [node for node in nx_graph.nodes if not bitstring_list[node]]
    S1 = [node for node in nx_graph.nodes if bitstring_list[node]]
    cut_edges = [(u, v) for u, v in nx_graph.edges if bitstring_list[u]!=bitstring_list[v]]
    uncut_edges = [(u, v) for u, v in nx_graph.edges if bitstring_list[u]==bitstring_list[v]]
    size_max_cut=len(cut_edges)
    return size_max_cut, [S0, S1], cut_edges, uncut_edges


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
