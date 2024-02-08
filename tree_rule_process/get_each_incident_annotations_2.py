import json
import joblib
import pandas as pd
from object_and_predicate_definition_1 import data_analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from num2words import num2words as n2w
import pickle as pk

# 这一份文件的目的是生成predicate、object的对应关系
# 主要是生成不同路径对应的状态命题，例如predicate=’State‘, subject=object='Normal'/'Abnormal'

if __name__=='__main__':
    #根据数据集选择对应的Decision_Tree_List模型,选择训练集文件
    target_dataset='Cardiotocography'
    dataset_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'

    #获得数据集的特征名称
    dataset=data_analysis(dataset_path)
    feature_names=dataset['attribute_names'][:-1]

    X_train=dataset['samples']
    y_train=dataset['labels']

    #特征名称作为object对象加入objects列表变量
    with open('./tree_rule_process/'+target_dataset+'/objects.json', 'r') as f:
        objects=json.load(f)

    with open('./tree_rule_process/'+target_dataset+'/predicates.json', 'r') as f:
        predicates=json.load(f)

    objects=objects+feature_names
    file_path='./Rules_based_on_Decision_Trees/'+target_dataset+'_Decision_Tree_List.pkl'
    #读取决策树模型文件
    Decision_Tree_List=joblib.load(file_path)
    n_decision_trees=len(Decision_Tree_List)#树模型的数量

    #读取树的路径和样本信息
    save_path='./tree_rule_process/'+target_dataset+'/tuple_right_path_and_sample.pk'
    with open(save_path, 'rb') as f:
        tuple_right_path_and_sample=pk.load(f)
    all_right_decision_path_dict_different_decision_trees,all_right_decision_path_correspond_to_sample_list_different_decision_trees=tuple_right_path_and_sample[0],tuple_right_path_and_sample[1]
    decision_path_and_its_annotation=[{} for _ in range(n_decision_trees)]#保存的是决策路径对应的annotation

    for index,decision_tree in enumerate(Decision_Tree_List):
        for path_key,decision_path in all_right_decision_path_dict_different_decision_trees[index].items():
            x_train=all_right_decision_path_correspond_to_sample_list_different_decision_trees[index][path_key][0]
            x_predict=decision_tree.predict([x_train])
            if(x_predict[0]==0.0):#如果预测结果=0.0，是正常样本，否则是异常样本
                predicate_name='State'
                subject_name,object_name='Normal','Normal'
            else:
                predicate_name='State'
                subject_name,object_name='Abnormal','Abnormal'
                #用于训练逻辑规则，只需要提取出对应的规则即可。但是要注意保留对应的训练样本，方便以后使用。
            decision_path_annotation={'decision_path':decision_path,'samples':all_right_decision_path_correspond_to_sample_list_different_decision_trees[index][path_key],"predicate":predicates.index(predicate_name),'subject':objects.index(subject_name),'object':objects.index(object_name)}
            # ---------解释 感觉必须要放完整的events进去，因为在find_rels的时候要根据events的实际信息查找告警告警之间的关系，以及查找object对象 ---------
            decision_path_and_its_annotation[index].setdefault(path_key,decision_path_annotation)

    with open('./tree_rule_process/'+target_dataset+'/annotations_decision_tree_train.pk','wb') as f:
        pk.dump(decision_path_and_its_annotation,f)

    with open('./tree_rule_process/'+target_dataset+'/annotations_decision_tree_train.pk','rb') as f:
        decision_path_and_its_annotation=pk.load(f)

    print(decision_path_and_its_annotation)

    # with open('./data/annotations_train.json','r') as f:
    #     eee=json.load(f)
    #     for key,anno in eee.items():
    #         print(key)
    #         print(len(anno['all_alarm_events']))
    #
