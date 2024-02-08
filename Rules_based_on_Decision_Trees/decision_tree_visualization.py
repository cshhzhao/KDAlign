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
    #Process the CSV file to obtain the input data in vector form and its corresponding labels
    read_file=pd.read_csv(dataset_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values

    labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

if __name__=='__main__':
    #Set dataset name
    target_dataset='Cardiotocography'
    
    dataset_test_path = './data/'+target_dataset+'/'+target_dataset+'_ML_Train_Preprocessed_Version.csv'
    dataset=data_preprocess(dataset_path=dataset_test_path)
    attribute_names=dataset['attribute_names']
    samples=dataset['samples']
    labels=dataset['labels']    

    #split datasets
    X_train, y_train = samples,labels    

    filename='./Rules_based_on_Decision_Trees/'+target_dataset+'_'+'Decision_Tree_List.pkl'
    DecisionTreeList=joblib.load(filename)

    for i_th_decision_tree in range(len(DecisionTreeList)):
        
        from sklearn import tree
        import dtreeviz
        import graphviz
        
        vis_tree = DecisionTreeList[i_th_decision_tree]
        viz_model = dtreeviz.model(vis_tree,
                                    X_train=X_train, y_train=y_train.astype('int'),
                                    feature_names=attribute_names[:-1],
                                    target_name=attribute_names[-1], class_names=["Normal", "Abnormal"])
        viz_figure = viz_model.view(scale=0.8)
        viz_figure.save('./Rules_based_on_Decision_Trees/'+target_dataset+'/visulaized_svg/'+target_dataset+'_visualized_'+str(i_th_decision_tree)+'_th_decision_tree'+'.svg')

    print(filename,'Visualization End！')
