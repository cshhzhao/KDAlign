import re

import pysat
import json
from copy import deepcopy

# with open('./tree_rule_process/Amazon/objects.json', 'r') as f:
#     objects=json.load(f)

# with open('./tree_rule_process/Amazon/predicates.json', 'r') as f:
#     predicates=json.load(f)

#_________________________________________________________
def triple2num(triple,var_pool):
    return var_pool.id(triple)

def num2triple(num, var_pool):
    return var_pool.obj(num)


def num2name(num, var_pool, pres, objs):
    p, s, o = num2triple(num, var_pool)
    return pres[p], objs[s], objs[o]


def name2num(pre, sub, obj, pres, objs, var_pool):
    num_predicate = len(pres)
    p, s, o = pres.index(pre), objs.index(sub), objs.index(obj)
    return triple2num((p, s, o), var_pool)


class Converter:
    def __init__(self, var_pool, pres, objs):
        self.var_pool = var_pool
        self.pres = pres
        self.objs = objs

    def triple2num(self, triple):
        return self.var_pool.obj2id[triple]

    def name2num(self, triple):
        pre, sub, obj = triple
        p, s, o = self.pres.index(pre), self.objs.index(sub), self.objs.index(obj)
        return self.triple2num((p, s, o))

    #
    # def graph2num(self,inci_graph,alarms_list,predicates,objs,fault_id):
    #     var_pool_ids=[]
    #
    #     for link_prop_triple in link_props_triple(inci_graph,alarms_list,predicates,objs,fault_id):
    #         var_pool_ids.append(self.triple2num(link_prop_triple))
    #     return var_pool_ids

    def path2num(self,current_decision_path_constraints):

        var_pool_ids=[]
        for constraint in current_decision_path_constraints:
            var_pool_ids.append(self.triple2num(constraint))
        return var_pool_ids

    def num2triple(self, num):
        return self.var_pool.id2obj[num]

    def num2name(self, num):
        pre, sub, obj = self.num2triple(num)
        return self.pres[pre], self.objs[sub], self.objs[obj]

class RelevantFormulaContainer:
    def __init__(self, r_clauses):
        #注意r_clauses是已经挑选过的相关子句了，不是[[[[ ]]]]
        #而是[[[ ]]]的结构
        self.r_clauses = r_clauses

        self.rvar_to_var = {}   #记录转化的第i个Formula是哪一个命题类型，里面的数字是通过变量池var_pool得到的
        self.var_to_rvar = {}   #记录了不同的命题类型是在第几次出现。方便对应回原本的三元组关系
        #这两个其实记录的是命题公式的最小单位的对应情况，比如var_pool中的(predicate,sub,obj)编辑为1，
        #那么这就是里面的最小单位，relevant_formula的类对象中重新进行了一下编号，便于使用

        literal_counter = 1
        for r_c in r_clauses:
            for l in r_c:
                if abs(l) not in self.var_to_rvar.keys():
                    self.var_to_rvar[abs(l)] = literal_counter
                    self.rvar_to_var[literal_counter] = abs(l)
                    literal_counter += 1

    def get_relevant_formula(self): #将子句转为合取范式CNF
        f = pysat.formula.CNF()
        for r_c in self.r_clauses:
            f.append([int(self.var_to_rvar[abs(l)] * abs(l) / l) for l in r_c])  #f的规则里面的命题id本身还是要取还原它本身对应的id，abs(l)/l的目的是还原这个子句的正负号
            #上述做的这个字典是帮助对应三元组关系，实际上里面规则的生成的时候id用的还是变量池中的id

        return f

    #目的是要得到原始的表达内容
    def get_original_repr(self, r_solution):
        return [int(self.rvar_to_var[abs(l)] * abs(l) / l) for l in r_solution]

def cnf_to_dimacs(file_name, clauses, num_atoms):
    with open(file_name,'w') as f:
        f.write(f'p cnf {num_atoms} {len(clauses)}\n')
        for c in clauses:
            for l in c:
                f.write(str(l) + ' ')
            f.write('0' + '\n')

def dimacs_to_cnf(file_name):
    num_atoms = None
    clauses = []
    for line in open(file_name, 'r'):
        if line[0] == 'p':
            num_atoms = int(line.split()[-2])
        elif line[0] == 'c':
            continue
        else:
            l = list(map(int, line.split()[:-1]))
            clauses.append(l)
    return clauses, num_atoms

def dimacs_to_nnf(file_name, c2d_path='./c2d_linux'):
    import os
    r = os.system(c2d_path + ' -in ' + file_name)
    return file_name + '.nnf', r