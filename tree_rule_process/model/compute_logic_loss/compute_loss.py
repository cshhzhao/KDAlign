from __future__ import division
from __future__ import print_function

import copy
from dgl.data import DGLDataset, load_graphs, save_graphs, split_dataset
from fault_scenario_dataset_13_small_sample_experiment.model.fault_relation_prediction.mydataset import FaultDataset,collate_fn_for_fault_train
from fault_scenario_dataset_13_small_sample_experiment.model.fault_relation_prediction.model_gat.gat import GAT
from torch.utils.data import DataLoader

import pickle as pk
import itertools
import sys


sys.path.append('../../')
sys.path.append('../pygcn/pygcn/')  #加载embedder模型的时候要在路径中放入model所在的文件件，否则会加载失败 注意！！！。会自动查找路径，这里相当于是给查找路径的时候提供了一种选择
import random
import time
import argparse
import os
import urllib.request
import json
from pysat import formula
import torch
import torch.multiprocessing
import numpy as np
import torch.nn.functional as F
from tools.incident_cluster_graphs import incident_graphs_dict
from model.misc.Conversion import Converter, POS_REL_NAMES_FULL, link_props_triple
from model.compute_logic_loss.loss_tools import load_data


clauses = pk.load(open('D:/LENSR_HWProject_version_beta/data/clauses.pk', 'rb'))
objs = pk.load(open('D:/LENSR_HWProject_version_beta/data/objects.pk', 'rb'))
pres = pk.load(open('D:/LENSR_HWProject_version_beta/data/predicates.pk', 'rb')) #这个pk文件是包括了
annotation = json.load(open('D:/LENSR_HWProject_version_beta/data/annotations_train.json'))
# annotation_test = json.load(open('../data/annotations_test.json'))
word_vectors = pk.load(open('D:/LENSR_HWProject_version_beta/data/word_vectors.pk', 'rb'))
tokenizers = pk.load(open('D:/LENSR_HWProject_version_beta/data/tokenizers.pk', 'rb'))
variables = pk.load(open('D:/LENSR_HWProject_version_beta/data/var_pool.pk', 'rb'))
var_pool = formula.IDPool(start_from=1)
for _, obj in variables['id2obj'].items():
    var_pool.id(obj)
converter = Converter(var_pool, pres, objs)
idx2filename = pk.load(open('D:/LENSR_HWProject_version_beta/data/vrd_raw/idx2incident_csn.pk', 'rb'))

embedder_filename = "vrd_ddnnf.reg0.1.ind.cls0.1.seed42.model" #因为直接用的torch.save(model)因此加载的时候把全部加载过来即可
indep_weights = 'ind' in embedder_filename
embedder = torch.load('D:/LENSR_HWProject_version_beta/model/pygcn/pygcn/model_save/' + embedder_filename)
node_features = pk.load(open('D:/LENSR_HWProject_version_beta/model/pygcn/pygcn/features.pk', 'rb'))['features']

def assignment_to_gcn_compatible(var_list, node_features):
    """
    :param var_list:
    :param rel_list:
    :return: adj, feature, label
    """
    features = torch.stack(
        [torch.FloatTensor(node_features['Global'])] +
        var_list +
        [torch.FloatTensor(node_features['And'])])
    labels = torch.FloatTensor([0] + [1] * len(var_list) + [3])
    adj = torch.eye(len(labels), len(labels))
    adj[0, :] = 1
    adj[:, 0] = 1
    adj[-1, :] = 1
    adj[:, -1] = 1

    r_inv = adj.sum(dim=1) ** (-1)
    r_mav_inv = torch.diag(r_inv)
    adj_normalized = torch.mm(r_mav_inv, adj)

    return adj_normalized, features, labels


#用于计算当前inci_csn对应的规则
def get_formula_from_incident(inci_csn,annotation,embedder,clauses,converter,objs,tokenizers):
    file_id=idx2filename.index(inci_csn) #idx2filename保存的是一个incident csn列表，incident csn所在第i个位置的i就是file id
    adj0, features0, labels0, idx_train0, idx_val0, idx_test0, _, _ = \
        load_data(filename=file_id, dataset='vrd_ddnnf', override_path='D:/LENSR_HWProject_version_beta/data/', and_or=False)
    adj0 = adj0.to_dense()
    features0 = features0
    labels0 = labels0
    output = embedder(features0.squeeze(0), adj0.squeeze(0), labels0)
    return output[0]


def prediction_to_assignment_embedding(softmax, embedder, tokenizers, pres, objs,incident_csn):
    """
    :param softmax: a torch.Tensor
    :param info: (inci_csn, label)
    :param embedders: (pe, ce, ae)
    :return: the embedding of assignment
    """

    def _feature(name):
        embedding = np.array([word_vectors[tokenizers['vocab2token'][i]] for i in name.split(' ')])
        summed_embedding = np.sum(embedding, axis=0)
        return summed_embedding #这里是得到节点的embedding

    embedded_clauses = []

    prop = softmax  #softmax是一个向量
    sub = 'subgraph'
    obj = 'subgraph'

    e_p = 0   #e_p意思不清楚
    for pres_idx in range(0, 27): #27的意思是有多少种故障类型
        e_p += prop[pres_idx] * torch.FloatTensor(_feature(tokenizers['token2vocab'][pres_idx]))  #不同的故障类型有不同的embedding

    e_p = (e_p + torch.FloatTensor(_feature(sub)) + torch.FloatTensor(_feature(obj))) / 3

    embedded_clauses.append(e_p) #目前找到的子句

     #这里是查找子图中告警与告警之间的关系

    incident_subgraph = incident_graphs_dict[int(incident_csn)]  # 获取当前incident对应的子结构
    current_pos_constraints = link_props_triple(incident_subgraph, annotation[incident_csn]['all_alarm_events'], pres, objs) #查找当前inci_csn对应的告警与告警之间的关系

    for alarm_pos_constraint in current_pos_constraints:
        #首先要根据id查找出他们对应的关系名称
        predicate_name=pres[alarm_pos_constraint[0]]
        if predicate_name in POS_REL_NAMES_FULL.keys():
            predicate_name=POS_REL_NAMES_FULL[predicate_name]
        sub = objs[alarm_pos_constraint[1]] #主语
        obj = objs[alarm_pos_constraint[2]] #宾语


        embedded_clauses.append((torch.FloatTensor(_feature(predicate_name)) +
                                 torch.FloatTensor(_feature(sub)) +
                                 torch.FloatTensor(_feature(obj))) / 3)

    adj0, features0, labels0 = assignment_to_gcn_compatible(embedded_clauses, node_features)

    embedded_clauses = embedder(features0.squeeze(0), adj0.squeeze(0), labels0)

    return embedded_clauses

#给定incident csn号 以及 对应的预测结果给出计算得到的logic loss
def logic_loss(inci_csn,prediction): #prediction要的是最后一层的输出，是27维的
    formula_embedding=get_formula_from_incident(inci_csn,annotation,embedder,clauses,converter,objs,tokenizers)

    #info指的就是当前的inci_csn是什么故障类型，(inci_csn,())
    #找到当前inci_csn对应的故障类型：
    info_fault_type=annotation[str(inci_csn)]["predicate"] #找到故障类型
    info=(str(inci_csn),info_fault_type)
    assignmen_embedding=prediction_to_assignment_embedding(F.softmax(prediction), embedder, tokenizers, pres, objs,inci_csn)
    loss_embedding=(formula_embedding-assignmen_embedding).norm()

    return loss_embedding
if __name__=='__main__':

    # GAT参数
    gnn_hidden_dim = 26  # GAT隐藏层参数
    gnn_num_layers = 2
    gnn_num_heads = 10
    mlp_hidden_dim = 89  # graph_embedding是89维向量，因此Logic formula的Embedding也是89
    mlp_num_layers = 5
    mlp_act_int = 0

    batch_size = 256

    # 首先是获取GAT模型的输入的Dataset对象
    data_split = [0.8, 0.1, 0.1]
    data = FaultDataset()
    dataset_train, dataset_val, dataset_test = split_dataset(data,
                                                             frac_list=data_split,
                                                             shuffle=False,
                                                             random_state=555)
    with open('D:/LENSR_HWProject_version_beta/model/Breast_Cancer_Wisconsin_relation_prediction/incident_in_order_with_dgl_bin.json', 'r') as f:
        inci_csns = json.load(f)  # 这里是用于索引当前训练的图像它所对应的告警与告警之间的关系，与annotation_train文件匹配，用于生成formula()因为annotation_train含有当前图结构蕴含的events信息

    # 生成训练集的数据加载器
    dataset_train_index = dataset_train.__len__()
    inci_csns_train = inci_csns[:dataset_train_index]
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        collate_fn=collate_fn_for_fault_train,
        drop_last=False,
        shuffle=True)

    dataset_val_index = dataset_train.__len__()
    inci_csns_val = inci_csns[dataset_train_index:dataset_train_index + dataset_val_index]
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        collate_fn=collate_fn_for_fault_train,
        drop_last=False,
        shuffle=True)

    # 1、加载预测模型
    NUM_MODEL = 1
    # make model
    if mlp_act_int == 0:
        mlp_act = F.relu
    else:
        mlp_act = F.tanh

    models = [GAT(num_node_types=data.num_node_types(), num_edge_types=data.num_edge_types(),
                  in_dim_node=gnn_hidden_dim, in_dim_edge=gnn_hidden_dim, hidden_dim=gnn_hidden_dim,
                  num_layers=gnn_num_layers, num_heads=[gnn_num_heads] * gnn_num_layers,
                  mlp_layers=[mlp_hidden_dim] * mlp_num_layers, mlp_act=mlp_act,
                  out_dim=data.num_label_classes()) for _ in range(NUM_MODEL)]

    for i in range(NUM_MODEL):
        models[i].load_state_dict(torch.load('D:/LENSR_HWProject_version_beta/model/Breast_Cancer_Wisconsin_relation_prediction/model_gat/gat_save/distilled_checkpoint.pt'))

    used_model = copy.deepcopy(models[0])  # 因为是Ensemble模型，只需要取其中一个模型就可以了

    total_train_inci = len(inci_csns_train)
    start_index = 0
    end_index = batch_size  # 因为要获取每次读入的所有的图结构对应的inci_csn号，用于获取其对应的规则
    for batched_g, batched_labels in dataloader_train:  # 开始遍历内容,注意不能使用tqdm，我也不知道为什么，反正用了就报错
        outs, g_embed = used_model(batched_g, batched_g.ndata['feat'], batched_g.edata['feat'])

        prediction_loss = F.cross_entropy(outs, batched_labels)  # 预测的loss

        outs_fault = []  # 存储了当前模型预测出来的故障类型，维度是[256,1]，最后长度不够256的时候以实际情况为准
        for out in outs.tolist():
            outs_fault.append(out.index(max(out)))

        if (end_index < total_train_inci):
            batched_inci_csns = inci_csns_train[start_index:end_index]
            start_index += batch_size  # 同时往后移batch size个位置
            end_index += batch_size
        else:
            batched_inci_csns = inci_csns_train[start_index:]

        logic_loss_value=logic_loss(str(batched_inci_csns[0]),outs[0]) #要保证输入的是str类型的变量
        print(logic_loss_value) #成功的找出logic loss的结果


