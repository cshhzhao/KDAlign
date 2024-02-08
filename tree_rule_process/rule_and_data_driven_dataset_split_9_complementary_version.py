import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from num2words import num2words as n2w
import pickle as pk

def data_analysis(dataset_name_path):#这里使用的是data_preprocess文件夹处理后的数据，label为 1或-1.0
    #处理csv文件，获得向量形式的输入数据，及其对应的标签
    read_file=pd.read_csv(dataset_name_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values
    if('Breast_Cancer_Wisconsin' in dataset_name_path):
        labels=read_file['diagnosis'].values
    else:
        labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

# 根据tuple_right_path_and_sample.pk文件保存的字典，划分Pima数据集。
if __name__=='__main__':
#———————————————————————————— 根据ADBench的Setting确定测试集样本数量占总样本的30%——————————————
#———————————————————————————— 训练集样本是去除掉样本后的数据 ————————————————————————————————
    #构造训练集，需要以原始数据集为基础
    target_dataset='Cardiotocography'
    dataset_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'

    #获得数据集的特征名称
    dataset=data_analysis(dataset_path)
    feature_names=dataset['attribute_names'][:-1]

    X_complete=dataset['samples']
    X_complete = [x.tolist() for x in X_complete]
    Y_complete=dataset['labels']
    Y_complete = [y.tolist() for y in Y_complete]
   
    num_total_samples=len(X_complete)

    num_rule_samples=0   
    filtered_samples=[]
    tuple_right_path_and_sample=pk.load(open('./tree_rule_process/'+target_dataset+'/tuple_right_path_and_sample.pk','rb'))
    #元组的第一个变量保存了决策树路径编号具体的路径值的列表，元组的第二个变量保存了决策树路径的编号及其对应的样本列表,我们需要的是第二个变量
    all_right_decision_path_dict_different_decision_trees, all_right_decision_path_correspond_to_sample_list_different_decision_trees=tuple_right_path_and_sample[0],tuple_right_path_and_sample[1]
    for decision_tree_index,decision_tree_path_key_and_samples in enumerate(all_right_decision_path_correspond_to_sample_list_different_decision_trees):
        for path_key,sample_list in decision_tree_path_key_and_samples.items():
            num_rule_samples+=len(sample_list)
            filtered_samples+=[sample.tolist() for sample in sample_list]

    X_ml_train=[]
    Y_ml_train=[]
    X_dropped_rule_samples=[] #dropped的样本只进行保存，但是不用于训练和测试
    Y_dropped_rule_samples=[]
    for index,x_train in enumerate(X_complete):
        if(x_train in filtered_samples):
            X_dropped_rule_samples.append(x_train)
            Y_dropped_rule_samples.append(Y_complete[index])
        else:
            X_ml_train.append(x_train)
            Y_ml_train.append(Y_complete[index])

    ML_Train_Evaluation_dataframe = pd.DataFrame(data=np.concatenate((np.array(X_ml_train),np.array(Y_ml_train).reshape((-1,1))),axis=1),columns=dataset['attribute_names'])
    ML_dropped_rule_samples_dataframe = pd.DataFrame(data=np.concatenate((np.array(X_dropped_rule_samples), np.array(Y_dropped_rule_samples).reshape((-1, 1))), axis=1),columns=dataset['attribute_names'])    

    ML_Train_Evaluation_dataframe_save_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Complementary_Evaluation_Version' #该文件包含Train_Preprocessed_Version数据集中剔除规则样本后的数据
    ML_dropped_rule_samples_dataframe_save_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Dropped_Rule_Samples' #该文件包含Train_Preprocessed_Version数据集中剔除的规则样本
    #YelpChi_ML_Train_Complementary_Evaluation_Version+YelpChi_ML_Train_Dropped_Rule_Samples=YelpChi_ML_Train_Preprocessed_Version 数据文件之间的关系

    ML_Train_Evaluation_dataframe.to_csv(ML_Train_Evaluation_dataframe_save_path+'.csv', index=False)
    ML_dropped_rule_samples_dataframe.to_csv(ML_dropped_rule_samples_dataframe_save_path+'.csv', index=False)      

    print('带有规则的训练集样本总数：',num_total_samples)
    print('剔除的规则样本总数：',len(X_dropped_rule_samples))
    print('剔除规则后用于Evaluation的训练集样本总数：',len(X_ml_train))    



# 1、规则定义清楚
# 2、Encode规则的方法的泛化性
# 3、规则存在错误的问题，kNN->smoothing.规则、样本连成图。

# if/else规则 树模型，能够进行encoding的方案。1、逻辑表达式；2、。。。
# Train数据集一半一半





