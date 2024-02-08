import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from num2words import num2words as n2w
import pickle as pk
from sklearn.inspection import DecisionBoundaryDisplay
import os, time, random

predicates=['Bigger','Small or Equal','State'] #所有的Tree Model存在且仅存在这2种关系

# Breast Cancer Wiscoin等数据集中如果有具体的特征名称那么咱们就用给定的名称，否则使用NLP进行描述，比如one\two\...
# object_to_full_en={'BloodPressure':'Blood Pressure',
#                    'SkinThickness':'Skin Thickness',
#                    'DiabetesPedigreeFunction':'Diabetes Pedigree Function'}

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

def get_sample_num_of_target_type(labels,target_type):
    
    target_sample_labels=[value for value in labels if(value==target_type)]
    return len(target_sample_labels)

#判断是否当前决策路径已经出现过错误决策
def is_in_filtered_path_list(current_path,filtered_path_list):
    for filtered_path in filtered_path_list:
        if(np.array_equal(current_path,filtered_path)):
            return True
    return False

#判断是否当前决策路径已经进行过决策并且还没有出错
def is_in_all_right_decision_path_dict(current_path,all_right_decision_path_dict:dict):
    #判断是否已经存在all_right的路径列表中，如果存在则返回对应的True以及对应的key值
    #如果不存在则返回False和-1即可
    for key,value in all_right_decision_path_dict.items():
        if(np.array_equal(current_path,value)):
            return True,key
    return False,-1


#判断是否当前决策路径属于过滤后的决策路径
def is_in_target_all_right_decision_path_dict(current_path_key,filtered_all_right_decision_path_keys):
    #判断是否已经存在all_right的路径列表中，如果存在则返回对应的True以及对应的key值
    #如果不存在则返回False和-1即可
    if(current_path_key in filtered_all_right_decision_path_keys):
        return True
    else:
        return False


#判断当前的样本是否出现在其他决策路径中，如果出现，则当前决策路径不对应当前样本，主要目的是保证每个样本只对应一条决策路径，这也是为了后续对齐任务，对比学习任务的可行性，否则一个样本对应多个决策路径，无法完成对比学习任务。
def exist_in_other_path_samples(all_right_decision_path_correspond_to_sample_list_different_DecisionTrees,x_train):
    x_train=x_train.tolist()
    for estimator_samples in all_right_decision_path_correspond_to_sample_list_different_DecisionTrees:
        for key, lists in estimator_samples.items():
            lists=[x.tolist() for x in lists]#要将里面的array转成list类型才能使用下面的操作
            if(x_train in lists):
                return True
    return False

# 保存一个objects文件，方便以后的加载
#这一份文件的目的
# 1、首先是要定义好命题逻辑中需要使用的谓词和对象，并保存为json文件
# 2、保存谓词和对象的同时，要保存对应的规则以及全部分类正确的样本，也就是要一一遍历样本在树模型下的决策路径。将决策路径存放到字典中，当一个决策路径对应的预测结果不正确时，将决策路径从样本中删除，否则路径的value列表加入样本值
# 3、根据已知的树的路径信息，添加对应的object值，构造规则数据，同时保存规则和样本的对应关系。

#保证每条路径都是完全正确的，这种情况下想要得到合适数量的决策路径数量比较困难
if __name__=='__main__':
    objects = ['Normal','Abnormal']#样本对象的State是正常还是异常
    #根据数据集选择对应的iForest模型
    target_dataset='Cardiotocography'
    dataset_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'

    #获得数据集的特征名称
    dataset=data_analysis(dataset_path)
    feature_names=dataset['attribute_names'][:-1]
    X_train=dataset['samples']
    y_train=dataset['labels']

    #特征名称作为object对象加入objects列表变量
    updated_feature_names=[]
    for feature_name in feature_names:
        
        #如果数据集的列名称是缩写，那么需要进行转换名称的指定，也就是使用如下注释后的代码
        # if feature_name in object_to_full_en.keys():
        #     updated_feature_names.append(object_to_full_en[feature_name])
        # else:
        #     updated_feature_names.append(feature_name)

        updated_feature_names.append(feature_name)        

    objects=objects+updated_feature_names
    
    #读取决策树模型文件
    selected_path_sample_num=5 #每条路径对应的异常样本数量至少大于当前变量值,参考Rules_based_on_Decision_Trees文件夹下的值
    filename='./Rules_based_on_Decision_Trees/'+target_dataset+'_'+'Decision_Tree_List.pkl'
    Decision_Tree_List=joblib.load(filename)
    n_decision_trees=len(Decision_Tree_List)#树模型的数量

    #--------------------------用于处理决策路径和样本的关系------------------------------------------------
    #定义保存的变量
    iteration_path_num=0#从0开始编码路径
    #由于ndarry类别不是hash类型的遍历，所以不可以用于dict的key值，为了让路径和样本之间能够统一，并且删除的方便，考虑使用路径编码对齐路径和样本的列表，参考如下操作
    all_right_decision_path_dict_different_DecisionTrees=[{} for _ in range(n_decision_trees)] #全部正确的决策路径列表，每个列表元素对应不同的树模型的路径字典
    all_right_decision_path_correspond_to_sample_list_different_DecisionTrees=[{} for _ in range(n_decision_trees)]#全部正确的决策路径列表，每个列表元素对应不同的树模型的路径字典

    filtered_decision_path_list=[[] for _ in range(n_decision_trees)] #存在错误的决策路径，这里不需要记录对应的样本
    # --------------------------------------------------------------------------

    for x_index,x_train in enumerate(X_train): #index的目的是查找对应的标签值
        for index,DecisionTree in enumerate(Decision_Tree_List):
            #存在一个路径是的样本x分类为真，则其他预测器就不考虑了
            x_decision_path=(DecisionTree.decision_path([x_train])).toarray()
            #base_estimator和iForest的predict结果有区别，这里我们只使用extraTreeRegressor的结果，predict的value>0.5认为是负类别，-1，否则认为是正类别1
            x_predict=DecisionTree.predict([x_train])

            is_in_filtered_path = is_in_filtered_path_list(x_decision_path, filtered_decision_path_list[index])
            is_in_all_right_decision_path,path_key=is_in_all_right_decision_path_dict(x_decision_path, all_right_decision_path_dict_different_DecisionTrees[index])
            if(x_predict==y_train[x_index] and x_predict==1.0):
                if(is_in_filtered_path): #如果存在错误的路径里面，那么当前路径和样本直接跳过即可
                    continue
                else:
                    if(is_in_all_right_decision_path):
                        if_x_train_exist_in_other_path_samples=exist_in_other_path_samples(all_right_decision_path_correspond_to_sample_list_different_DecisionTrees,x_train)
                        if(if_x_train_exist_in_other_path_samples==False):
                            all_right_decision_path_correspond_to_sample_list_different_DecisionTrees[index][path_key].append(x_train)
                    else:
                        if_x_train_exist_in_other_path_samples = exist_in_other_path_samples(all_right_decision_path_correspond_to_sample_list_different_DecisionTrees, x_train)
                        if(if_x_train_exist_in_other_path_samples==False):
                            all_right_decision_path_dict_different_DecisionTrees[index].setdefault(iteration_path_num,x_decision_path)
                            all_right_decision_path_correspond_to_sample_list_different_DecisionTrees[index].setdefault(iteration_path_num,[x_train])
                            iteration_path_num+=1#路径编码更新，+1
                    break
            else:
                if(is_in_filtered_path):
                    continue
                else:
                    if(is_in_all_right_decision_path):
                        all_right_decision_path_dict_different_DecisionTrees[index].pop(path_key)
                        all_right_decision_path_correspond_to_sample_list_different_DecisionTrees[index].pop(path_key)
                        filtered_decision_path_list[index].append(x_decision_path)
                    else:
                        filtered_decision_path_list[index].append(x_decision_path)

    target_anomalous_samples_count=0
    sample_num_list=[]
    target_all_right_decision_path_correspond_to_sample_list_different_DecisionTrees=[] #这里保存的是全部正确的路径中
    for DecisionTree_samples in all_right_decision_path_correspond_to_sample_list_different_DecisionTrees:
        temp_count = 0
        DecisionTree_paths_and_samples={}
        for key, lists in DecisionTree_samples.items():
            if(len(lists)>=selected_path_sample_num):
                temp_count += len(lists)
                sample_num_list.append(temp_count)
                DecisionTree_paths_and_samples.setdefault(key,lists)
        target_anomalous_samples_count += temp_count
        target_all_right_decision_path_correspond_to_sample_list_different_DecisionTrees.append(DecisionTree_paths_and_samples)


    target_path_count=0
    target_all_right_decision_path_dict_different_DecisionTrees=[]
    for index,DecisionTree_paths in enumerate(all_right_decision_path_dict_different_DecisionTrees):
        DecisionTree_decision_path_dict={}
        for path_key, decision_path in DecisionTree_paths.items():
            is_in_all_right_decision_path=is_in_target_all_right_decision_path_dict(path_key, list(target_all_right_decision_path_correspond_to_sample_list_different_DecisionTrees[index].keys()))
            if(is_in_all_right_decision_path): #如果当前路径在过滤后的完全正确的路径里面出现，那么就统计当前的决策路径，否则不统计
                target_path_count+=1
                DecisionTree_decision_path_dict.setdefault(path_key,decision_path)
        target_all_right_decision_path_dict_different_DecisionTrees.append(DecisionTree_decision_path_dict)

    total_anomalous_sample_num=get_sample_num_of_target_type(labels=y_train,target_type=1.0) #数据集中异常样本的数量，根据label对象处理就可以了
    covered_percentage_of_anomalous_samples=target_anomalous_samples_count/total_anomalous_sample_num #决策树路径覆盖的异常样本数量占总异常样本的比例

    #接下来就是根据树模型和对应的路径信息生成object对象
    for index, DecisionTree in enumerate(Decision_Tree_List):
        decision_tree_threshold = DecisionTree.tree_.threshold
        for p_key,decision_path in target_all_right_decision_path_dict_different_DecisionTrees[index].items():
            threshold_value_index=[index for index,bool_sign in enumerate(decision_path[0]) if(bool_sign==1)]
            threshold_value_list=[n2w(round(v,2)) for v in decision_tree_threshold[threshold_value_index][:-1]]
            # if('thousand,' in threshold_value_list):
            #     print(111)
            updated_threshold_value_list=[]
            for v in threshold_value_list:
                if(',' in v):
                    v=v.split(',')
                    new_v=''
                    for n_v in v:
                        new_v+=n_v
                else:
                    new_v=v
                updated_threshold_value_list.append(new_v)
            objects+=updated_threshold_value_list

    with open('./tree_rule_process/'+target_dataset+'/objects.json', 'w') as f:
        json.dump(objects, f)

    with open('./tree_rule_process/'+target_dataset+'/predicates.json', 'w') as f:
        json.dump(predicates, f)

    with open('./tree_rule_process/'+target_dataset+'/attribute_names.json', 'w') as f:
        json.dump(updated_feature_names, f)


    print('针对异常的全部分类正确的决策路径对应的样本数量(覆盖的异常样本数量)：',target_anomalous_samples_count)
    print('总异常样本数量：',total_anomalous_sample_num)
    print('覆盖的异常样本数占总异常样本的比例：',covered_percentage_of_anomalous_samples)
    print('针对异常的全部分类正确的决策路径数量:', target_path_count)
    tuple_right_path_and_sample=(target_all_right_decision_path_dict_different_DecisionTrees,target_all_right_decision_path_correspond_to_sample_list_different_DecisionTrees)
    save_path='./tree_rule_process/'+target_dataset+'/tuple_right_path_and_sample.pk'
    with open(save_path, 'wb') as f:
        pk.dump(tuple_right_path_and_sample,f)

