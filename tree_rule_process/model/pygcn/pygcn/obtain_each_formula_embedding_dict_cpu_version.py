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
from utils import load_data
from torch.autograd import Variable
# from model.pygcn.pygcn.models import GCN #MLP是输出0,1的，这里只需要GCN即可
import copy

sys.path.append('./pygcn/pygcn/')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#2、加载logic embeder模型
embedder_filename = "vrd_ddnnf.reg0.1.ind.cls0.1.seed42_no_val.model" #因为直接用的torch.save(model)因此加载的时候把全部加载过来即可
indep_weights = 'ind' in embedder_filename

#torch.load()上报ModuleNotFoundError: No module named 'models'错误的原因是当前的工作路径和保存模型之前的路径不一致所导致的
embedder = torch.load('./tree_rule_process/model/pygcn/pygcn/model_save/vrd_ddnnf.reg0.1.ind.cls0.1.seed42_no_val.model',map_location='cpu')

node_features = pk.load(open('./tree_rule_process/model/pygcn/pygcn/features.pk', 'rb'))['features']

list_name=os.listdir(r'./tree_rule_process/Amazon/vrd_ddnnf')
count=4
total=0
idx2filename = pk.load(open(r'./tree_rule_process/Amazon/vrd_raw/idx2incident_csn.pk', 'rb'))
embedding_dict={}
for f in list_name:
    try:
        var=f[:-4] #名称
        var_int=int(var) #第i个故障类型的embedding
        adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0=load_data(var, 'vrd_ddnnf', and_or=True, override_path=r'./tree_rule_process/Amazon')

        adj0=adj0.to_dense()
        embedding=embedder(features0.squeeze(0), adj0.squeeze(0), labels0.squeeze(0))
        embedding_pooling=embedding.mean(dim=0)
        embedding_dict.setdefault(idx2filename[var_int],embedding_pooling)
        total+=1
        count=4
    except:
        continue

pk.dump(embedding_dict, open(f'./tree_rule_process/data/Amazon//formula_embedding_dict.pk', 'wb'))
pk.dump(embedding_dict, open(f'./tree_rule_process/Amazon/formula_embedding_dict.pk', 'wb'))

# if __name__=='__main__':
#     e_d=pk.load(open(f'./tree_rule_process/Amazon/formula_embedding_dict.pk', 'rb'))
#     a=torch.zeros([1,64],dtype=float)
#
#     for key,value in e_d.items():
#         a=a+0.2*value
#
#     print(a)