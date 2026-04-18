# Controlling Continuous Relaxation for Combinatorial Optimization

A **PyTorch implementation** of the NeurIPS 2024 paper:  
🔗 [Controlling Continuous Relaxation for Combinatorial Optimization](https://nips.cc/virtual/2024/poster/92998)

> ## 👉 Looking for the latest version? Use [**QQA4CO**](https://github.com/Yuma-Ichikawa/QQA4CO).
>
> [QQA4CO](https://github.com/Yuma-Ichikawa/QQA4CO) (ICLR 2025) is the actively
> maintained successor: it generalises CRA into a unified gradient-based
> annealing framework, ships a **PyTorch Geometric** re-implementation of
> CRA-PI-GNN as an optional backend (`pip install "qqa[pignn]"`) that runs on
> modern GPUs (e.g. NVIDIA Blackwell B200 / `sm_100`) where DGL prebuilt
> wheels are not yet available, and adds spin glasses, TSP, coloring, a
> Streamlit GUI, and a CLI.
>
> **This repository (CRA4CO) is preserved verbatim as the canonical
> NeurIPS 2024 release** for exact paper reproduction. For new projects,
> reach for QQA4CO first. See the [Related work](#related-work) section
> below for a side-by-side decision table.

## Abstract

Unsupervised Learning (UL)-based solvers for **Combinatorial Optimization (CO)** train neural networks to generate soft solutions by directly optimizing the CO objective using **continuous relaxation strategies**. While these solvers offer advantages over traditional methods, they suffer from:

1️⃣ **Optimization Issues** – Getting trapped in local optima 🔄  
2️⃣ **Rounding Issues** – Artificial rounding from continuous to discrete spaces weakens robustness ⚠️  

To overcome these, we propose **Continuous Relaxation Annealing (CRA)** – a **rounding-free learning method** for UL-based solvers. CRA dynamically adjusts a penalty term, transitioning from smoothing non-convexity to enforcing discreteness, eliminating artificial rounding. 🏆

💡 **Key Benefits:**  
✅ Significantly boosts UL-based solver performance  
✅ Outperforms existing UL-based methods & greedy algorithms  
✅ Eliminates artificial rounding  
✅ Accelerates the learning process 🚀

---

## Installation

This package was implemented with **Python 3.11.11**. To install dependencies, run:

```bash
pip install -r requirements.txt
```

### 📦 Dependencies:
✅ **dgl** → `2.1.0`  
✅ **torch** → `2.4.0`  
✅ **numpy** → `1.26.4`  
✅ **pandas** → `2.2.2`  
✅ **matplotlib** → `3.10.0`  
✅ **seaborn** → `0.13.2`  
✅ **scikit-learn** → `1.6.1`  
✅ **networkx** → `3.4.2`  
✅ **tqdm** → `4.67.1`  

---

## License
This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE.txt) for details.

---

## 🚀 Usage Guide

### **Step 1: Setup the Environment & Load the Problem**

```python
import random
import os
import copy
from collections import OrderedDict, defaultdict
from itertools import chain, islice, combinations
from time import time
from tqdm import tqdm

import dgl
import torch
import numpy as np
import networkx as nx

from src import utils, gnn, instance

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Fix seed for reproducibility
SEED = 0
utils.fix_seed(SEED)
torch_type = torch.float32
```

#### Define Graph & Problem Parameters
```python
N, d, p, graph_type = 5000, 20, None, "reg"
nx_graph = nx.random_regular_graph(d=d, n=N, seed=SEED)
dgl_graph = dgl.from_networkx(nx_graph).to(device)
Q_mat = utils.qubo_dict_to_torch(nx_graph, instance.gen_q_dict_mis_sym(nx_graph, penalty=2)).to(device)
```

---

### **Step 2: Define the GNN Model & Embeddings**

```python
in_feats = int(dgl_graph.number_of_nodes()**(0.5))
hidden_size = in_feats
num_class = 1
dropout = 0.0

model = gnn.GCN_dev(in_feats, hidden_size, num_class, dropout, device).to(device)
embedding = nn.Embedding(dgl_graph.number_of_nodes(), in_feats).type(torch_type).to(device)
```

---

### **Step 3: Define the Loss Function**

```python
def loss(probs, reg_param, curve_rate=2):    
    probs_ = torch.unsqueeze(probs, 1)
    cost = (probs_.T @ Q_mat @ probs_).squeeze()
    reg_term = torch.sum(1 - (2 * probs_ - 1) ** curve_rate)
    return cost + reg_param * reg_term, cost, reg_term
```

---

### **Step 4: Train the PI-GNN Solver**

```python
num_epoch = int(1e5)
lr = 1e-4
weight_decay = 1e-2
tol = 1e-4
patience = 1000
check_interval = 1000
curve_rate = 2

model, bit_string_PI, cost, reg_term, runtime = gnn.fit_model(
    model, dgl_graph, embedding, loss,
    num_epoch=num_epoch, lr=lr, weight_decay=weight_decay,
    tol=tol, patience=patience, device=device,
    annealing=False, init_reg_param=0,
    annealing_rate=0, check_interval=check_interval, curve_rate=curve_rate
)
```

---

### **Step 5: Train the CRA-PI-GNN Solver**

```python
init_reg_param = -20
annealing_rate = 1e-3

model, bit_string_CRA, cost, reg_term, runtime = gnn.fit_model(
    model, dgl_graph, embedding, loss,
    num_epoch=num_epoch, lr=lr, weight_decay=weight_decay,
    tol=tol, patience=patience, device=device,
    annealing=True, init_reg_param=init_reg_param,
    annealing_rate=annealing_rate, check_interval=check_interval,
    curve_rate=curve_rate
)
```

---

### **Step 6: Evaluate the Results**

```python
size_mis_CRA, _, number_violation = utils.postprocess_gnn_mis(bit_string_CRA, nx_graph)
size_mis_PI, _, number_violation = utils.postprocess_gnn_mis(bit_string_PI, nx_graph)
print(f"Independent set size: (CRA) {size_mis_CRA.item()}, (PI) {size_mis_PI.item()}")
```

**Expected Output:**
```
Independent set size: (CRA) 853, (PI) 0
```

---

## Related work

For a unified, gradient-based annealing framework that subsumes CRA and
extends it to spin glasses (Ising, Edwards–Anderson, SK), statistical-physics
problems (binary perceptron, Hopfield), categorical / permutation problems
(Coloring, TSP, QAP), and ships a Streamlit dashboard, see:

> Y. Ichikawa, Y. Arai. **"Continuous Tensor Relaxation for Finding Diverse
> Solutions in Combinatorial Optimization."** *International Conference on
> Learning Representations (ICLR), 2025.*
> Code: <https://github.com/Yuma-Ichikawa/QQA4CO> ·
> Paper: <https://openreview.net/forum?id=9EfBeXaXf0>

QQA4CO ships a **PyTorch Geometric** re-implementation of CRA-PI-GNN as an
**optional backend** (`pip install qqa[pignn]`), so users on newer GPU
architectures (e.g. NVIDIA Blackwell B200 / `sm_100`) that DGL's prebuilt
wheels do not yet target can still reproduce the CRA results. The reference
DGL implementation in *this* repository is preserved verbatim as the
canonical NeurIPS 2024 release.

| Use case                                                          | Recommended repo                                  |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| Reproduce the NeurIPS 2024 CRA-PI-GNN paper exactly (DGL stack)   | **CRA4CO** (this repo)                            |
| Run CRA-PI-GNN on Blackwell GPUs without DGL                      | **QQA4CO** with `qqa[pignn]`                      |
| Apply the broader QQA framework to spin glasses / TSP / coloring  | **QQA4CO**                                        |
| Use a GUI / Streamlit dashboard                                   | **QQA4CO** (`qqa gui`)                            |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{
ichikawa2024controlling,
title={Controlling Continuous Relaxation for Combinatorial Optimization},
author={Yuma Ichikawa},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=ykACV1IhjD}
}
```

**Now you're ready to experiment with Continuous Relaxation Annealing!** Happy Researching!