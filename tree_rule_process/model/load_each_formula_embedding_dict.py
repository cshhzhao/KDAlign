import os
import time
from tqdm import tqdm
import torch
import dgl
import numpy as np
import json
import torch.nn.functional as F
import sys
import pickle as pk
import matplotlib.pyplot as plt
from tree_rule_process.model.pygcn.pygcn.utils import load_data
from torch.autograd import Variable
# from model.pygcn.pygcn.models import GCN #MLP是输出0,1的，这里只需要GCN即可
import copy

sys.path.append('./pygcn/pygcn/')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


embedding_dict=pk.load(open(f'../Breast_Cancer_Wisconsin/formula_embedding_dict.pk', 'rb'))
print(embedding_dict)
