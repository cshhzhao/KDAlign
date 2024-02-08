import subprocess
from tqdm import tqdm
import pysat
from pysat.formula import CNF,WCNF
from pysat.solvers import Solver
import numpy as np

from model.misc.Conversion import triple2num,num2triple

def neg(a):
    out=[]
    for i in range(len(a)):
        out.append(int(-a[i])) #改成非命题
    return out

def rand_assign(nv):
    out=[]
    for i in range(nv):
        sign=np.random.choice([-1,1])
        out.append(int((i+1)*sign))
    return out

def find_truth(f,n,solver_name='g3',assumptions=[]):
    #f指的是有子句构成的formula，里面是一个一个的命题，n是查找可满足的解的个数
    out=[]
    s=Solver(name=solver_name)
    s.append_formula(f.clauses)
    #参考的demo，cnf的含义是（-1 析取 2)合取 3
    # with Solver(bootstrap_with=[[-1, 2], [3]]) as s:
    #     for m in s.enum_models():
    #         print(m)
    for idx , m in enumerate(s.enum_models(assumptions=assumptions)):#这里查找的是枚举所有是的cnf表达式为真的事实，其实就是assignments
        out.append(m)
        if idx>=n-1:
            break
    s.delete()
    return out

def find_false(f, n, max_try=100, solver_name='g3'):
    if n == 0:
        return []
    nv = f.nv
    out = []
    s = Solver(name=solver_name)
    s.append_formula(f.clauses)
    tries = 0
    while len(out) < n and tries < max_try:
        assign = rand_assign(nv)
        if s.solve(assumptions=assign) is False:
            out.append(assign)
        tries += 1
    s.delete()
    return out


def find(f, n, solver_name='g4', assumptions=[]):
    truth = find_truth(f, n, solver_name=solver_name, assumptions=assumptions)
    false = find_false(f, len(truth), solver_name=solver_name)
    return truth, false


def cnf_to_dimacs(file_name, clauses, num_atoms):
    with open(file_name,'w') as f:
        f.write(f'p cnf {num_atoms} {len(clauses)}\n')  #f前缀的目的是支持python在内的表达式
        for c in clauses:
            if type(c) != int:
                for l in c:
                    f.write(str(l) + ' ')
                f.write('0' + '\n')
            else:
                for cc in clauses:
                    f.write(str(cc) + ' ')
                f.write('0' + '\n')
                break


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


def dimacs_to_nnf(dimacs_path, nnf_path, c2d_path='c2d_linux', ):
    import os
    r, output = subprocess.getstatusoutput(c2d_path + ' -in ' + dimacs_path) #linux和windows都是这个命令
    # os.system('move ' + dimacs_path+'.nnf' + ' ' + './tree_rule_process/Amazon/vrd_ddnnf_raw')
    # os.system('move ' + dimacs_path+'.nnf' + ' ' + nnf_path)
    os.system('mv ' + dimacs_path+'.nnf' + ' ' + nnf_path) #linux的mv命令
    return output, r

if __name__ == "__main__":

    f1 = CNF()
    f1.append([-1, 2])

    print(f1.clauses)
    print(find(f1, 5))






