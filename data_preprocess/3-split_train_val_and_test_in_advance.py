import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle as pk

def data_analysis(dataset_name_path):#这里使用的是data_preprocess文件夹处理后的数据，label为 1或-1.0
    #处理csv文件，获得向量形式的输入数据，及其对应的标签
    read_file=pd.read_csv(dataset_name_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values

    labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

# 根据tuple_right_path_and_sample.pk文件保存的字典，划分Pima数据集。
if __name__=='__main__':
    #构造训练集，需要以原始数据集为基础
    Target_dataset_path = './data/Amazon.csv'

    #获得数据集的特征名称
    dataset=data_analysis(Target_dataset_path)
    X_complete=dataset['samples']
    X_complete = [x.tolist() for x in X_complete]
    Y_complete=dataset['labels']
    Y_complete = [y.tolist() for y in Y_complete]

    test_size=0.3 # 按照ADBench的比例去划分。70%train，30%test。但是这里，咱们70%的train是要改变的，先进行preprocess训练规则以及获取规则对应的样本，而后得到train_evaluzation_version

    X_train_random, X_test_random, Y_train_random, Y_test_random = train_test_split(X_complete, Y_complete, test_size=test_size, random_state=42)

    ML_Train_dataframe_random = pd.DataFrame(data=np.concatenate((np.array(X_train_random),np.array(Y_train_random).reshape((-1,1))),axis=1),columns=dataset['attribute_names'])
    ML_Test_dataframe_random = pd.DataFrame(data=np.concatenate((np.array(X_test_random), np.array(Y_test_random).reshape((-1, 1))), axis=1),columns=dataset['attribute_names'])

    ML_Train_dataframe_random_save_path = './data//Amazon//Amazon_ML_Train_Preprocessed_Version' #这里面的样本还没有经过规则的过滤
    ML_Test_dataframe_random_save_path = './data//Amazon//Amazon_ML_Test_Evaluation_Version'
    ML_Train_dataframe_random.to_csv(ML_Train_dataframe_random_save_path+'.csv', index=False)
    ML_Test_dataframe_random.to_csv(ML_Test_dataframe_random_save_path+'.csv', index=False)