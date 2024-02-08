
import sys

import joblib

sys.path.append('.')
sys.path.append('../')

import json
import pickle as pk
import copy
from tree_rule_process.model.misc.Conversion import Converter
from tree_rule_process.object_and_predicate_definition_1 import data_analysis
import pysat
from tqdm import tqdm
from num2words import num2words as n2w
from numerizer import numerize #对应的版本是0.1.5
import numerizer

from tree_rule_process.model.misc.Formula import find,find_truth

def obtain_decision_path_constraints(decision_path,decision_tree,attribute_names,objs,pres):
    decision_path_constraints=[]
    decision_tree_feature=decision_tree.tree_.feature
    decision_tree_threshold=decision_tree.tree_.threshold
    #上述读取的是决策树模型的参数

    #decision_path存储的是一个list变量，其结构和decision_tree_feature、decision_tree_threshold相呼应，意思是index是一样的
    #feature作为subject，threshold作为object，节点后面跟左孩子或右孩子作为predicate。
    #由于decision path的list使用的是DFS深度搜索结构，所以如果一个元素后面的值为0说明接下来查找的是右孩子，则predicate为Bigger；否则为Small or Equal
    for tree_node_index,value in enumerate(decision_path[0]):
        if(value==1):
            if(decision_tree_threshold[tree_node_index]==-2.0):
                return decision_path_constraints#如果阈值是-2.0说明到达了叶子节点直接返回
            leaf_attribute_name=attribute_names[decision_tree_feature[tree_node_index]]
            leaf_str_format_threshold=n2w(round(decision_tree_threshold[tree_node_index],2))
            temp_leaf_str_format_threshold=''
            if(',' in leaf_str_format_threshold):
                temp=leaf_str_format_threshold.split(',')
                for v in temp:
                    temp_leaf_str_format_threshold+=v
                leaf_str_format_threshold=temp_leaf_str_format_threshold
            sub_id=objs.index(leaf_attribute_name)
            obj_id=objs.index(leaf_str_format_threshold)
            if(decision_path[0][tree_node_index+1]==0):
                rel_id=pres.index('Bigger')
            else:
                rel_id = pres.index('Small or Equal')
            decision_path_constraints.append((rel_id, sub_id, obj_id))
    return decision_path_constraints

if __name__=='__main__':
    target_dataset='Cardiotocography'

    annotations=pk.load(open('./tree_rule_process/'+target_dataset+'/annotations_decision_tree_train.pk','rb'))

    objs=json.load(open('./tree_rule_process/'+target_dataset+'/objects.json')) #注意名称的映射
    pres=json.load(open('./tree_rule_process/'+target_dataset+'/predicates.json'))
    attribute_names=json.load(open('./tree_rule_process/'+target_dataset+'/attribute_names.json','r'))

    file_path = './Rules_based_on_Decision_Trees/'+target_dataset+'_Decision_Tree_List.pkl'

    # 读取决策树模型文件
    Decision_Tree_List = joblib.load(file_path)
    n_decision_trees = len(Decision_Tree_List)  # 树模型的数量

    rel_to_pos={}#总共有两个key值，分别是State(Normal.Normal),State(Abnormal,Abnormal)。该字典存储了结果和对应的不同前提条件
    clauses=[] #存放所有的决策路径定义的所有命题


    var_pool=pysat.formula.IDPool(start_from=1)
    converter = Converter(var_pool, pres, objs)

    deal_with_incident=0
    for tree_index, annos in enumerate(annotations):#开始遍历描述
        for path_key, decision_path_annotation in annos.items():#每一条路径要生成一个命题公式
            # decision_path_annotation = {'decision_path': decision_path, 'samples':sample_list,"predicate": predicates.index(predicate_name),'subject': objects.index(subject_name), 'object': objects.index(object_name)}
            #
            current_decision_path_constraints=obtain_decision_path_constraints(decision_path_annotation['decision_path'],Decision_Tree_List[tree_index],attribute_names,objs,pres) #返回一个列表，存放若干个三元组(rel_id, subject_id, object_id))三元组，每一个三元组代表了当前决策树中间节点的值。

            rel_id=decision_path_annotation['predicate'] #里面放的是id
            subject_id=decision_path_annotation['subject'] #描述中额subject和object只有subgraph一种类型，其余的subject和object需要从决策树路径里面寻找
            object_id=decision_path_annotation['object']

            if (rel_id, subject_id, object_id) not in rel_to_pos.keys():
                rel_to_pos[(rel_id, subject_id, object_id)] = []

            #这里key=正常或异常，value是一组list，代表了不同的规则
            for constraint in current_decision_path_constraints:
                if constraint not in rel_to_pos[(rel_id, subject_id, object_id)]:
                    rel_to_pos[(rel_id, subject_id, object_id)].append(constraint)  # 添加的是list不需要循环
            # rel_to_pos[(rel_id, subject_id, object_id)].append(current_decision_path_constraints) #添加的是list不需要循环

    print(rel_to_pos)

    # #循环结束后rel_to_pos字典中key是异常类型，key的value是一个list，list中存放的元素仍然是list，list中存放的是一个一个的三元组
    for rel,constraints_list in rel_to_pos.items():
        this_clauses = [var_pool.id(rel)]

        for constraint in constraints_list:  #列表中是一个一个的元素
            #因为Constraint是一个list，此外由于这里的蕴含具有合取关系
            #子句存储的设计也要变化
            this_clauses.append(-var_pool.id(constraint))

        clauses.append(this_clauses)

    print(rel_to_pos)

    pk.dump({'rel':list(rel_to_pos.keys())}, open('./tree_rule_process/'+target_dataset+'/rels_train.pk', 'wb'))
    pk.dump(clauses, open('./tree_rule_process/'+target_dataset+'/clauses_train.pk', 'wb'))
    pk.dump({'obj2id': dict(var_pool.obj2id), 'id2obj': dict(var_pool.id2obj)},
            open('./tree_rule_process/'+target_dataset+'/var_pool_train.pk', 'wb'))
    pk.dump(pres, open('./tree_rule_process/'+target_dataset+'/predicates.pk', 'wb'))
    pk.dump(objs, open('./tree_rule_process/'+target_dataset+'/objects.pk', 'wb'))  #注意存储的pk文件是个list，存放的是object_full_name


