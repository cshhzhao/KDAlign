import os,sys

import joblib

sys.path.append('..')
from itertools import combinations_with_replacement
from find_rels_3 import obtain_decision_path_constraints
from tqdm import tqdm
import pickle as pk
import json
import pysat
from pysat import formula
from model.misc.Conversion import Converter, RelevantFormulaContainer
from model.misc.Formula import cnf_to_dimacs, find

#clause中的第一个元素就是异常类型，只需要匹配 这一项即可
def relevant_clauses(clauses,predicate,converter):
    for c in clauses:
        for d_clause in c: #d_clause本身已经是id了 因为c是一个析取式
            problem_relation_id=abs(d_clause)#这里找出来的是故障类型的id
            problem_relation_triple=converter.num2triple(problem_relation_id)
            if(predicate==problem_relation_triple[0]):#如果predicate
                return c

def prepare_clauses(clauses,anno,converter,objs,pres,predicate_id,current_decision_path_constraints):
    # predicate=anno['predicate'] #其实就是state_id

    r_clauses=relevant_clauses(clauses,
                               predicate_id,
                               converter)


    #原因出在assumption！！！
    assumptions=[]

    anno_var_pool_ids=converter.path2num(current_decision_path_constraints)

    #注意这里得到的是关系，关系不需要重复 注意！
    anno_var_pool_ids=list(set(anno_var_pool_ids)) #去重复

    for anno_var_pool_id in anno_var_pool_ids:
        assumptions.append([anno_var_pool_id])


    #这里更新好的r_clauses相当于是[[故障a对应的所有的约束关系],[当前图结构中真实存在的告警之间的关系]]
    r_clauses = [r_clauses] + assumptions  # r_clauses是predicate的关系，是一个list的拼接操作
    r_container = RelevantFormulaContainer(r_clauses)  #注意r_clauses是[[],[],[],...]的结构

    return r_clauses, r_container

    
target_dataset='Cardiotocography'

clauses = pk.load(open('./tree_rule_process/'+target_dataset+'/clauses_train.pk', 'rb'))
objs = pk.load(open('./tree_rule_process/'+target_dataset+'/objects.pk', 'rb'))
pres = pk.load(open('./tree_rule_process/'+target_dataset+'/predicates.pk', 'rb'))
# annotation = json.load(open('./tree_rule_process/'+target_dataset+'/annotations_train.json'))
annotations = pk.load(open('./tree_rule_process/'+target_dataset+'/annotations_decision_tree_train.pk','rb'))
word_vectors = pk.load(open('./tree_rule_process/'+target_dataset+'/word_vectors.pk', 'rb'))
tokenizers = pk.load(open('./tree_rule_process/'+target_dataset+'/tokenizers.pk', 'rb'))
variables = pk.load(open('./tree_rule_process/'+target_dataset+'/var_pool_train.pk', 'rb'))

attribute_names = json.load(open('./tree_rule_process/'+target_dataset+'/attribute_names.json', 'r'))

file_path = './Rules_based_on_Decision_Trees/'+target_dataset+'_Decision_Tree_List.pkl'

# 读取决策树模型文件
Decision_Tree_List = joblib.load(file_path)
n_decision_trees = len(Decision_Tree_List)  # 树模型的数量

var_pool = formula.IDPool(start_from=1)
for _, obj in variables['id2obj'].items():
    var_pool.id(obj)

converter = Converter(var_pool, pres, objs)  # 猜测这里多定义了一个变量池的目的是不扰乱从文件读入的变量池子，因此做了备份

decision_path_list=[] #里面存放的是二元组，第一个元素是decision tree num，第二个元素是决策路径的path_key
idx = 0 #这个记录的是路径编号

if not os.path.exists('./tree_rule_process/'+target_dataset+'/vrd_raw/'): #创建逻辑图结构关系的原始数据
    os.mkdir('./tree_rule_process/'+target_dataset+'/vrd_raw/')

for tree_index, annos in enumerate(annotations):  # 开始遍历描述
    for path_key, decision_path_annotation in annos.items():  # 每一条路径要生成一个命题公式
        current_decision_path_constraints = obtain_decision_path_constraints(decision_path_annotation['decision_path'], Decision_Tree_List[tree_index], attribute_names, objs,pres)  # 返回一个列表，存放若干个三元组(rel_id, subject_id, object_id))三元组，每一个三元组代表了当前决策树中间节点的值。
        decision_path_list.append((tree_index,path_key))
        predicate_id = decision_path_annotation['predicate']

        r_clauses,r_container=prepare_clauses(clauses,decision_path_annotation,converter,objs,pres,predicate_id,current_decision_path_constraints)
        num_atom = max([abs(j) for i in r_container.get_relevant_formula().clauses for j in i])

        cnf_to_dimacs(f'./tree_rule_process/Cardiotocography/vrd_raw/{idx}.cnf', r_container.get_relevant_formula().clauses,num_atom)
        r_sol_t, r_sol_f = find(r_container.get_relevant_formula(), 5, assumptions=[])
        for sol_t_idx in range(len(r_sol_t)):
            cnf_to_dimacs(f'./tree_rule_process/Cardiotocography/vrd_raw/{idx}.st{sol_t_idx}.cnf',
                            [[i] for i in r_sol_t[sol_t_idx]], num_atom)
            cnf_to_dimacs(f'./tree_rule_process/Cardiotocography/vrd_raw/{idx}.sf{sol_t_idx}.cnf',
                            [[i] for i in r_sol_f[sol_t_idx]], num_atom)
        idx+=1
pk.dump(decision_path_list, open(f'./tree_rule_process/'+target_dataset+'/vrd_raw/idx2incident_csn.pk', 'wb'))