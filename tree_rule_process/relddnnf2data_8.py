import os, sys

import joblib

sys.path.append('../')
import json
import pickle as pk

import numpy as np
from tqdm import tqdm
from pysat import formula
from pathos.multiprocessing import ProcessingPool as Pool
from find_rels_3 import obtain_decision_path_constraints

from model.misc.Formula import dimacs_to_cnf
from model.misc.Conversion import Converter
from rel2cnf_5 import prepare_clauses

"""
This script converts d-DNNF generated from stanford VRD dataset to pygcn compatible data.
Note that it is different from ddnnf2data, because we have a 'relevant clause' concept when dealing with stanford vrd.
"""

clauses = pk.load(open('./tree_rule_process/Cardiotocography/clauses_train.pk', 'rb'))
objs = pk.load(open('./tree_rule_process/Cardiotocography/objects.pk', 'rb'))
pres = pk.load(open('./tree_rule_process/Cardiotocography/predicates.pk', 'rb'))
annotations = pk.load(open('./tree_rule_process/Cardiotocography/annotations_decision_tree_train.pk','rb'))
word_vectors = pk.load(open('./tree_rule_process/Cardiotocography/word_vectors.pk', 'rb'))
tokenizers = pk.load(open('./tree_rule_process/Cardiotocography/tokenizers.pk', 'rb'))
variables = pk.load(open('./tree_rule_process/Cardiotocography/var_pool_train.pk', 'rb'))
var_pool = formula.IDPool(start_from=1)

attribute_names = json.load(open('./tree_rule_process/Cardiotocography/attribute_names.json', 'r'))

file_path = './Rules_based_on_Decision_Trees/'+'Cardiotocography_Decision_Tree_List.pkl'

# 读取决策树模型文件
Decision_Tree_List = joblib.load(file_path)
n_decision_trees = len(Decision_Tree_List)  # 树模型的数量

for _, obj in variables['id2obj'].items():
    var_pool.id(obj)
converter = Converter(var_pool, pres, objs)


idx2filename = pk.load(open('./tree_rule_process/Cardiotocography/vrd_raw/idx2incident_csn.pk', 'rb'))


def _feature_leaf(num):
    name = list(converter.num2name(abs(num)))
    # for i in range(len(name)):
    #     if name[i] in POS_REL_NAMES_FULL.keys():
    #         name[i] = POS_REL_NAMES_FULL[name[i]]
    embedding = np.array(
        [word_vectors[tokenizers['vocab2token'][i]] for i in ' '.join([name[1], name[0], name[2]]).split(' ')])
    summed_embedding = np.sum(embedding, axis=0) / 3
    return summed_embedding if num > 0 else -summed_embedding


def write_data(input_file, output_file, features):
    ddnnf = open(input_file, 'r')
    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    and_children = open(output_file[2], 'w')
    or_children = open(output_file[3], 'w')

    relations_str = ''
    variables_str = ''

    and_children_list = []
    or_children_list = []

    num_feature = 50

    feature_OR = features['Or']
    feature_AND = features['And']
    feature_G = features['Global']

    file_id = input_file.split('/')[-1].split('.')[-3] if '.s' in input_file else input_file.split('/')[-1].split('.')[-2]
    print(file_id)
    file_id = int(file_id)

    cnf, _ = dimacs_to_cnf(input_file.replace('vrd_ddnnf_raw', 'vrd_raw').replace('nnf', 'cnf'))

    predicate_id=annotations[idx2filename[file_id][0]][idx2filename[file_id][1]]['predicate']

    # def prepare_clauses(clauses, anno, converter, objs, pres, predicate_id, current_decision_path_constraints):
    current_decision_path_constraints = obtain_decision_path_constraints(annotations[idx2filename[file_id][0]][idx2filename[file_id][1]]['decision_path'],Decision_Tree_List[idx2filename[file_id][0]], attribute_names, objs,pres)  # 返回一个列表，存放若干个三元组(rel_id, subject_id, object_id))三元组，每一个三元组代表了当前决策树中间节点的值。
    _, r_container = prepare_clauses(clauses, annotations[idx2filename[file_id][0]][idx2filename[file_id][1]], converter, objs,pres,predicate_id,current_decision_path_constraints)
    cnf = [r_container.get_original_repr(c) for c in cnf]

    # add global var
    feature = feature_G
    label = 0
    variables_str += str(0) + '\t'
    for j in range(len(feature)):
        variables_str += str(feature[j]) + '\t'
    variables_str += str(label) + '\n'

    line_num = -1
    for line in ddnnf.readlines():
        if line_num > -1:
            line = line.split()
            type = line[0]
            children = line[1:]
            if type == 'L':
                feature = _feature_leaf(r_container.get_original_repr([int(children[0])])[0])
                label = 1  # leaf node
            elif type == 'O':
                feature = feature_OR
                label = 2  # OR node
                or_children_list.append([])
                for child in children[2:]:
                    child = int(child)
                    or_children_list[-1].append(child + 1)
                    relations_str += str(child + 1) + '\t' + str(line_num + 1) + '\n'
            elif type == 'A':
                feature = feature_AND
                label = 3  # AND node
                and_children_list.append([])
                for child in children[1:]:
                    child = int(child)
                    and_children_list[-1].append(child + 1)
                    relations_str += str(child + 1) + '\t' + str(line_num + 1) + '\n'

            variables_str += str(line_num + 1) + '\t'
            for j in range(len(feature)):
                variables_str += str(feature[j]) + '\t'
            variables_str += str(label) + '\n'

        line_num += 1

    # add edge for global variable
    for j in range(line_num):
        relations_str += str(j + 1) + '\t' + str(0) + '\n'

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()

    json.dump(and_children_list, and_children)
    json.dump(or_children_list, or_children)


#--------------------------------------------如果使用超线程注意要在main函数外面定义变量！！！！-----------------
features = pk.load(open('./tree_rule_process/model/pygcn/pygcn/features.pk', 'rb'))['features']

directory_in_str = './tree_rule_process/Cardiotocography/vrd_ddnnf_raw/'
directory_in_str_out = './tree_rule_process/Cardiotocography/vrd_ddnnf/'

if __name__ == '__main__':

    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)


    def _worker(file):
        if file.endswith(".nnf"):
            input_dire = os.path.join(directory_in_str, file)
            output_dire = [os.path.join(directory_in_str_out, file[:-4] + '.var'),
                           os.path.join(directory_in_str_out, file[:-4] + '.rel'),
                           os.path.join(directory_in_str_out, file[:-4] + '.and'),
                           os.path.join(directory_in_str_out, file[:-4] + '.or')]
            can_go_next = True
            for f in output_dire:
                can_go_next = can_go_next and os.path.exists(f)
            if can_go_next:
                return
            write_data(input_dire, output_dire, features)


    with Pool() as p:
        for _ in tqdm(p.imap(_worker, os.listdir(directory_in_str)), total=len(os.listdir(directory_in_str))):
            pass

    # for f in tqdm(os.listdir(directory_in_str), total=len(os.listdir(directory_in_str))): #单线程，用于调试
    #     _worker(f)