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

#0、确定将模型加载到哪一个设备中
selected_device_id=0

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def cuda_input(adj, features, labels):
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    return adj, features, labels

#1、选择对应的数据集
target_dataset='Cardiotocography'

#2、加载logic embeder模型
embedder_filename = "vrd_ddnnf.reg0.1.ind.cls0.1.seed42_no_val.model" #因为直接用的torch.save(model)因此加载的时候把全部加载过来即可
indep_weights = 'ind' in embedder_filename

#torch.load()上报ModuleNotFoundError: No module named 'models'错误的原因是当前的工作路径和保存模型之前的路径不一致所导致的；注意gpu训练的模型保存时会记录gpu id，在加载时候为了避免设备编号的冲突，先将模型映射到cpu中
embedder = torch.load('./tree_rule_process/model/pygcn/pygcn/model_save/'+target_dataset+'/vrd_ddnnf.reg0.1.ind.cls0.1.seed42_no_val.model',map_location='cpu')
#将设备从cpu转移至当前程序指定的gpu中训练
embedder.cuda(0)

list_name=os.listdir('./tree_rule_process/'+target_dataset+'/vrd_ddnnf')

itered_var_and_files={}

total=0
idx2filename = pk.load(open('./tree_rule_process/'+target_dataset+'/vrd_raw/idx2incident_csn.pk', 'rb'))
embedding_dict={}
for f in list_name:
    try:
        var=f[:-4] #名称
        var_int=int(var) #第i个故障类型的embedding
        if('and' in f): #500.and => var_int=500, 500.or ==> var_int=50，会导致重复验证的情况，所以这里要确定是and后缀的文件才执行embedder的前向传播过程

            # #调试代码
            # if(var_int not in list(itered_var_and_files.keys())):
            #     itered_var_and_files.setdefault(var_int,[f])
            # else:
            #     itered_var_and_files[var_int].append(f)
            #     print(itered_var_and_files[var_int])

            adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0=load_data(var, 'vrd_ddnnf', and_or=True, override_path='./tree_rule_process/'+target_dataset)
            adj0, features0, labels0=cuda_input(adj0, features0, labels0)
            adj0=adj0.to_dense()
            embedding=embedder(features0.squeeze(0), adj0.squeeze(0), labels0.squeeze(0)).to('cpu')
            embedding_pooling=embedding.mean(dim=0)
            embedding_dict.setdefault(idx2filename[var_int],embedding_pooling)
            total+=1
            count=4
        else:
            continue
    except:
        continue
print('总共获取了',len(list(embedding_dict.keys())),'个规则的Embeddings')
pk.dump(embedding_dict, open('./data/'+target_dataset+'/formula_embedding_dict.pk', 'wb'))
pk.dump(embedding_dict, open('./tree_rule_process/'+target_dataset+'/formula_embedding_dict.pk', 'wb'))

# if __name__=='__main__':
#     e_d=pk.load(open(f'./tree_rule_process/Cardiotocography/formula_embedding_dict.pk', 'rb'))
#     a=torch.zeros([1,64],dtype=float)
#
#     for key,value in e_d.items():
#         a=a+0.2*value
#
#     print(a)