import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from dgl.data import FraudYelpDataset,FraudAmazonDataset
from num2words import num2words as n2w
import pandas as pd
import os


def Amazon_Dataloader():
    dataset = FraudAmazonDataset()
    graph = dataset[0]
    num_classes = dataset.num_classes
    feat = graph.ndata['feature']
    label = graph.ndata['label']

    print('Class Num:', num_classes)

    samples_and_labels=np.concatenate((feat,label.reshape((-1,1))),axis=1)
    column_names=[]
    for i in range(1,len(samples_and_labels[0])+1):
        column_names.append(n2w(i))
    column_names[-1]='Outcome'
    dataset=pd.DataFrame(samples_and_labels,columns=column_names)
    print(os.getcwd()) 
    # CN: 注意Linux系统下 代码执行过程中的路径信息和windows下有所不同，要注意区分，最好是查看一下当前路径
    # EN: Note that the path information under the code execution process under Linux system is different from Windows, and it is best to check the current path to pay attention to the distinction
    #/home/user_name/MyWork/KDAlign
    print(os.path.exists('./data'))
    dataset.to_csv('./data/Amazon.csv',index=False)

    return dataset,feat,label

if __name__=='__main__':
    # x, edge_index, y, train_mask, valid_mask, test_mask=Amazon_Dataloader(datapath=r'../dataset/Amazon/Amazon.mat')
    dataset, features, labels=Amazon_Dataloader()
    print()
