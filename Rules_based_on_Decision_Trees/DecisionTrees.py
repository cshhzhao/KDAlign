import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import roc_auc_score

def data_preprocess(dataset_path):
    #处理csv文件，获得向量形式的输入数据，及其对应的标签
    read_file=pd.read_csv(dataset_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values

    labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

if __name__=='__main__':
    #Set Dataset Name
    target_dataset='Cardiotocography'

    dataset_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'
    dataset=data_preprocess(dataset_path=dataset_path)
    attribute_names=dataset['attribute_names']
    samples=dataset['samples']
    labels=dataset['labels']

    #划分数据集
    X_train, y_train = samples,labels

    selected_criterion='gini'
    selected_splitter='random'
    max_features=1
    random_state=None
    max_depth=10 #  Set the depth of Decision Tree models, avoiding too long rules

    Decision_Tree_Classifier_Num=10 # Set the num of Decision Tree Models for acquiring rules
    Decision_Tree_List=[] # For saving DT model objects.
    for i_th_decision_tree in range(Decision_Tree_Classifier_Num):
        #训练Decision Trees模型`
        DecisionTree = DecisionTreeClassifier(criterion=selected_criterion,splitter=selected_splitter,max_features=max_features,random_state=random_state,max_depth=max_depth)
        
        DecisionTree.fit(X_train,y_train)

        Decision_Tree_List.append(DecisionTree)

        # ———————————————— Optional Operations : Visualize each trained decision tree models ——————————————————————————————————
        # If you need, plead execture `crtl + /` (i.e., key shorcut of  Visual Studio Code) to cancel the code comment in the following 
        from sklearn import tree
        import dtreeviz
        import graphviz
        
        # # Visualization Start
        # vis_tree = DecisionTree
        # viz_model = dtreeviz.model(vis_tree,
        #                             X_train=X_train, y_train=y_train.astype('int'),
        #                             feature_names=attribute_names[:-1],
        #                             target_name=attribute_names[-1], class_names=["Normal", "Abnormal"])
        # viz_figure = viz_model.view(scale=0.8)
        # viz_figure.save('./Rules_based_on_Decision_Trees/'+target_dataset+'/'+target_dataset+'_visualized_'+str(i_th_decision_tree)+'_th_decision_tree'+'.svg')

        # for x_index, x_train in enumerate(X_train):  # index的目的是查找对应的标签值
        #     x_decision_path=(DecisionTree.decision_path([x_train])).toarray()
        #     # If the prediction result >0.5, then anomaly sample. Otherwise, normal samples.
        #     x_predict=DecisionTree.predict([x_train]) #output detection results

        # #Extracting tree structures from models.
        # #Leverage DFS search to iterature tree structures.
        # vis_tree_children_left=vis_tree.tree_.children_left
        # vis_tree_children_right=vis_tree.tree_.children_right
        # vis_tree_feature=vis_tree.tree_.feature
        # vis_tree_threshold=vis_tree.tree_.threshold
        # vis_tree_impurity=vis_tree.tree_.impurity
        # vis_tree_n_node_samples=vis_tree.tree_.n_node_samples
        # vis_tree_value=vis_tree.tree_.value
        
        # # Showing the statistical attribute value of DT models.
        
        # print("children_left:", vis_tree_children_left)
        # print("children_right:", vis_tree_children_right)
        # print("feature:", vis_tree_feature)
        # print("threshold:", vis_tree_threshold)
        # print("impurity:", vis_tree_impurity)
        # print("n_node_samples:", vis_tree_n_node_samples)
        # print("value:", vis_tree_value)
        # # Visualization End
    
    #save decision tree models as ./pkl files
    filename='./Rules_based_on_Decision_Trees/'+target_dataset+'_'+'DecisionTree_List.pkl'
    joblib.dump(Decision_Tree_List, filename)

    filename='./Rules_based_on_Decision_Trees/'+target_dataset+'_'+'DecisionTree.pkl'
    DecisionTreeList=joblib.load(filename)
