import os, sys
sys.path.append('../')
import argparse

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from model.misc.Formula import dimacs_to_nnf

in_dir = f'./tree_rule_process/Cardiotocography/vrd_raw/'
out_dir = f'./tree_rule_process/Cardiotocography/vrd_ddnnf_raw/'
# #-------------------------------------------
if __name__=='__main__':
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    def _worker(f):
        if f.endswith('.cnf'):
            # output, r = dimacs_to_nnf(in_dir + f, out_dir + f[:-4] + '.nnf', c2d_path='./tree_rule_process/c2d_linux')
            output, r = dimacs_to_nnf(in_dir + f, out_dir, c2d_path='./tree_rule_process/c2d_linux')
            if 'nan' in output or r != 0:
                print(f'Failed {f}')

    # with Pool() as p: #多线程操作
    #     for _ in tqdm(p.uimap(_worker, os.listdir(in_dir)), total=len(os.listdir(in_dir))):
    #         pass

    for f in tqdm(os.listdir(in_dir), total=len(os.listdir(in_dir))):#单线程操作，用于调试
        _worker(f)
