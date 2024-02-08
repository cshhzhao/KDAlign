import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import os, time, random

import warnings
warnings.filterwarnings("ignore")


def data_preprocess(dataset_path):
    read_file=pd.read_csv(dataset_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values

    labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

def get_sample_num_of_target_type(labels,target_type):
    
    target_sample_labels=[value for value in labels if(value==target_type)]
    return len(target_sample_labels)

#Determine whether the current decision path has made an erroneous decision or whether the identified sample is normal.
def is_in_filtered_path_list(current_path,filtered_path_list):
    for filtered_path in filtered_path_list:
        if(np.array_equal(current_path,filtered_path)):
            return True
    return False

#Determine whether the current decision path has been decided and has not been errored, and the predictions are all abnormal samples
def is_in_all_right_decision_path_dict(current_path,all_right_decision_path_dict:dict):
    #Determine whether the path list of `all_right` already exists. If it exists, return the corresponding True and the corresponding key value.
    #If it does not exist, return False and -1
    for key,value in all_right_decision_path_dict.items():
        if(np.array_equal(current_path,value)):
            return True,key
    return False,-1

#Determine whether the current decision path belongs to the filtered decision path
def is_in_target_all_right_decision_path_dict(current_path_key,filtered_all_right_decision_path_keys):
    if(current_path_key in filtered_all_right_decision_path_keys):
        return True
    else:
        return False
    
def exist_in_other_path_samples(all_right_decision_path_correspond_to_sample_list_different_DecisionTrees,x_train):
    x_train=x_train.tolist()
    for DecisionTree_samples in all_right_decision_path_correspond_to_sample_list_different_DecisionTrees:
        for key, lists in DecisionTree_samples.items():
            lists=[x.tolist() for x in lists]#要将里面的array转成list类型才能使用下面的操作
            if(x_train in lists):
                return True
    return False

def sub_Decision_Tree_auto_training_process_with_specific_saved_strategy(i_th_process,target_dataset,X_train,y_train,max_depth=10, max_features=1,n_decision_trees=10,random_state=None,criterion='gini',splitter='random',selected_path_sample_num=10,specific_saved_num=None):
    
    print('Start executing ',i_th_process,'-th thread，',' the tuned hyperparameters are: ','max_depth=',max_depth,'max_features=',max_features,' Total training',n_decision_trees,'decision tree models')


    Decision_Tree_list=[]
    for i in range(n_decision_trees):
        DecisionTree = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_features=max_features,random_state=random_state,max_depth=max_depth)
    
        DecisionTree.fit(X_train,y_train)        

        Decision_Tree_list.append(DecisionTree)    

    #--------------------------Search for all right decision paths and corresponding samples and their quantities-----------------------------------------------
    iteration_path_num=0
    all_right_decision_path_dict_different_DecisionTrees=[{} for _ in range(n_decision_trees)] #全部正确的决策路径列表，每个列表元素对应不同的树模型的路径字典
    all_right_decision_path_correspond_to_sample_list_different_DecisionTrees=[{} for _ in range(n_decision_trees)]#全部正确的决策路径列表，每个列表元素对应不同的树模型的路径字典

    filtered_decision_path_list=[[] for _ in range(n_decision_trees)]
    # --------------------------------------------------------------------------

    for x_index,x_train in enumerate(X_train):
        for index,DecisionTree in enumerate(Decision_Tree_list):
            x_decision_path=(DecisionTree.decision_path([x_train])).toarray()
            x_predict=DecisionTree.predict([x_train])

            is_in_filtered_path = is_in_filtered_path_list(x_decision_path, filtered_decision_path_list[index])
            is_in_all_right_decision_path,path_key=is_in_all_right_decision_path_dict(x_decision_path, all_right_decision_path_dict_different_DecisionTrees[index])
            if(x_predict==y_train[x_index] and x_predict==1.0):
                if(is_in_filtered_path):
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
                            iteration_path_num+=1
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
            if(is_in_all_right_decision_path): 
                target_path_count+=1
                DecisionTree_decision_path_dict.setdefault(path_key,decision_path)
        target_all_right_decision_path_dict_different_DecisionTrees.append(DecisionTree_decision_path_dict)
 
    total_anomalous_sample_num=get_sample_num_of_target_type(labels=y_train,target_type=1.0)
    covered_percentage_of_anomalous_samples=target_anomalous_samples_count/total_anomalous_sample_num
    
    if(specific_saved_num is not None):
        if(target_path_count>=specific_saved_num):

            filename='./Rules_based_on_Decision_Trees/'+target_dataset+'/'+target_dataset+'_'+'Decision_Tree_List_'+str(i_th_process)+'.pkl'
            joblib.dump(Decision_Tree_list, filename)        

            print('—————————————————————————— \n'
                'Saving Decision Tree Models (Satisfactory): \n',
                'Finishing executing ',i_th_process,'-th thread:\n',
                'Model File Name:', filename,
                'Hyperparameters is: ','max_depth=',max_depth,'max_features=',max_features,' Training total',n_decision_trees,' decision tree models',
                'Total anomaly samples: ',total_anomalous_sample_num,'Rate: ',covered_percentage_of_anomalous_samples,
                'Rule-detect anomaly samples ', target_path_count,
                '——————————————————————————')

            return  

        else:
            print('—————————————————————————— \n',
                'Finishing executing',i_th_process,'-th thread; Unsatisfactory Decision Tree Models. \n',
                '——————————————————————————\n')
            return
            
    else:
        #保存训练的模型到dataset_name+DecisionTree.pkl文件
        filename='./Rules_based_on_Decision_Trees/'+target_dataset+'/'+target_dataset+'_'+'Decision_Tree_List_'+str(i_th_process)+'.pkl'
        joblib.dump(Decision_Tree_list, filename)        

        print('—————————————————————————— \n'
            'Saving Decision Tree Models (Satisfactory): \n',
            'Finishing executing ',i_th_process,'-th thread:\n',
            'Model File Name:', filename,
            'Hyperparameters is: ','max_depth=',max_depth,'max_features=',max_features,' Training total',n_decision_trees,' decision tree models',
            'Total anomaly samples: ',total_anomalous_sample_num,'Rate: ',covered_percentage_of_anomalous_samples,
            'Rule-detect anomaly samples ', target_path_count,
            '——————————————————————————')        

        return            

if __name__=='__main__':
    target_dataset='Cardiotocography'

    dataset_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'
    dataset=data_preprocess(dataset_path=dataset_path)
    attribute_names=dataset['attribute_names']
    samples=dataset['samples']
    labels=dataset['labels']

    anomalous_samples_num=get_sample_num_of_target_type(labels,target_type=1.0)

    #split datasets
    X_train, y_train = samples,labels
    
    DecisionTree_Parameter_Tuple_list=[
                                        #  Cardiotocography
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 'auto'),
                                       (10, 9),
                                       (10, 9),
                                       (10, 9),
                                       (10, 9),       
                                       (10, 9),
                                       (10, 9),
                                       (10, 9),
                                       (10, 9),     
                                        #  Cardiotocography

                                        #  Amazon                                    
                                        # (10, 10),
                                        # (10, 10),
                                        # (10, 10),
                                        # (10, 10),
                                        # (10, 10),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),       
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),
                                        # (10, 15),       
                                        #  Amazon         
                                        #                                     
                                      ] #(elemen_1=max_depth, element_2=max_features)。
        
    process_num=len(DecisionTree_Parameter_Tuple_list) 
    
    pool=Pool(24)

    n_decision_trees=5 # decision tree number

    #debgu version
    for i in range(process_num):
        if((i+1)==len(DecisionTree_Parameter_Tuple_list)):
            break
        
        sub_Decision_Tree_auto_training_process_with_specific_saved_strategy(
                                                                             i,
                                                                             target_dataset,
                                                                             X_train,
                                                                             y_train,
                                                                             DecisionTree_Parameter_Tuple_list[i][0],
                                                                             DecisionTree_Parameter_Tuple_list[i][1],
                                                                             n_decision_trees,
                                                                             None,
                                                                             'gini',
                                                                             #   'random',
                                                                             'best',                        
                                                                             5,
                                                                             None
                                                                            ) #第二个传入的参数是调用函数的参数    

    print("———————— start ————————") 
    pool.close()
    pool.join()
    print("———————— end ————————")                              