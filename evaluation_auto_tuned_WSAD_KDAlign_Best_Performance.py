# ------------Core files---------------

# -*- coding: utf-8 -*-
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch import Tensor
import torch
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, CrossEntropyLoss,Softmax,BCELoss,BCEWithLogitsLoss
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk
import ot  # An optimal transport library from https://pythonot.github.io/
import ot.plot
import os
import copy
import numpy as np
from semi_supervised_model_save_and_accuracy import Saved_Module_Based_On_Val
import argparse
import itertools
import pandas as pd
import random
from model.feawad import FeaWAD
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Leverage Sinkhorn OT Algorithm

def data_preprocess_no_noise(dataset_path):
    #处理csv文件，获得向量形式的输入数据，及其对应的标签

    read_file=pd.read_csv(dataset_path)
    attribute_names=list(read_file.keys())
    samples=read_file[read_file.keys()[:-1]].values

    labels = read_file['Outcome'].values

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}

def data_preprocess_wsad(dataset_path,labeled_anomaly_samples=10):
    read_file=pd.read_csv(dataset_path).astype('float32')
    attribute_names=list(read_file.keys())    
    
    data_normal_samples=np.array([single for single in read_file.values[:,:] if(single[-1]==0.0)])
    data_abnormal_samples=np.array([single for single in read_file.values[:,:] if(single[-1]==1.0)])

    np.random.shuffle(data_normal_samples)
    np.random.shuffle(data_abnormal_samples)
    
    normal_num=data_normal_samples.shape[0] #正常样本数量
    abnormal_num=data_abnormal_samples.shape[0]

    wsad_abnormal_num = labeled_anomaly_samples

    wsad_abnormal_samples = data_abnormal_samples[:wsad_abnormal_num, :]

    remaining_abnormal_samples = data_abnormal_samples[wsad_abnormal_num:, :]

    remaining_abnormal_samples[:, -1] = 0

    data_wsad_samples = np.concatenate((wsad_abnormal_samples, remaining_abnormal_samples, data_normal_samples), axis=0)
    np.random.shuffle(data_wsad_samples)    
    
    samples=data_wsad_samples[:, :-1].astype('float32')
    labels = data_wsad_samples[:, -1].astype('float32')

    return {'attribute_names':attribute_names,'samples':samples,'labels':labels}    

class SampleDataset(Dataset):
    # load the dataset
    def add_gaussian_noise(data, mean, std_dev):
        noise = np.random.normal(mean, std_dev, data.shape) 
        noisy_data = data + noise 
        return noisy_data

    def __init__(self, X, y, add_noise=False, mean = 0, std_dev = 0 ):
        self.X = X
        self.y = y
        self.mean = mean
        self.std_dev = std_dev
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # if add noise to dataset
        if(add_noise):
            self.X = SampleDataset.add_gaussian_noise(self.X, mean ,std_dev)
        print(self.X.shape)
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]        

def score_computation(labels,probs): #Referred the `def eval(self, labels, probs):` in jianheng的NeurIPS 2023 GADBench: https://github.com/squareRoot3/GADBench/blob/master/models/detector.py
    score = {}
    labels=labels.reshape((-1))
    probs=probs.reshape((-1))
    with torch.no_grad():
        score['AUC_ROC'] = roc_auc_score(labels, probs)
        score['AUCPR'] = average_precision_score(labels, probs)
        labels = np.array(labels)
        k = int(labels.sum())
    score['RECK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
    return score    

def obtain_maximum_figure_num(figure):
    if(figure<1):
        dot_num=0
        temp=1
        while(figure<temp):
            dot_num+=1
            temp=temp*0.1
        if(str(figure) in str(temp*10.0)):
            return -1*(dot_num-1)
        return -1*dot_num
    else:
        dot_num=1
        temp=1
        while(figure>=temp):
            dot_num+=1
            temp=temp*10
        return dot_num-1
    


def train_model(train_dl,val_dl,test_dl, model,save_module,epoch=100,lr=0.01):
    # define the optimization

    consecutive_no_improvement = 0

    optimizer = Adam(model.net.parameters(), lr=lr)
    # enumerate epochs
    for epoch in range(epoch):
        # enumerate mini batches
        current_train_loss = 0.
        for i_th_batch, (batch_x, batch_y) in enumerate(train_dl):

            pred, sub_result, sample_2_embeddings = model.forward(batch_x,batch_y)

            # calculate loss
            loss = model.criterion(batch_y, pred, sub_result)

            # clear the gradients            
            optimizer.zero_grad()
            # credit assignment            
            loss.backward()
            
            current_train_loss += loss.data

            # update model weights
            optimizer.step()
            # print("epoch: {}, batch: {}, loss: {}".format(epoch, i_th_batch, loss.data))
        
        print("epoch: {}, average_loss: {}".format(epoch, current_train_loss/(i_th_batch+1))) 
        train_score, train_loss = None, None
        val_score, _,_ = evaluate_model_on_target_dataloader(val_dl, model) 

        #Save model performance based on f1@k included in val_score
        save_module.update_model_and_loss(current_train_loss=train_loss,current_model=model,current_train_score=train_score,current_val_score=val_score)                      
        if consecutive_no_improvement >= save_module.early_stop_num:
            print(f"Early stopping at epoch {epoch + 1} as there was no improvement in {save_module.early_stop_num} epochs.")
            break
        elif val_score['RECK'] > save_module.best_val_reck:
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
    test_score, _,_ = evaluate_model_on_target_dataloader(test_dl, model)     
    save_module.update_model_best_test_result(best_test_score=test_score)    
    
    return

def evaluate_model_on_target_dataloader(evaluate_data,model):
    probs, actuals = [], []
    sample_embeddings=[]
    loss = -1.0
    embeddings = None
    with torch.no_grad():
        for i_th_batch, (batch_x, batch_y) in enumerate(evaluate_data):

            _, scores = model.inference_score_forward(batch_x)

            # retrieve numpy array
            scores = scores.detach().to('cpu').numpy()
            actual = batch_y.flatten()
            # actual = actual.reshape((len(actual), 1))
            # store
            probs.append(scores) #save the probability prediction vectors
            actuals.append(actual)
        i_th_batch +=1                      
                          
    probs, actuals = vstack(probs), vstack(actuals)
    score=score_computation(labels=actuals,probs=probs)
    return score,sample_embeddings,(loss/i_th_batch)

def KDAlign_FeaWAD_train_model(train_dl,val_dl,test_dl,model,rule_embeddings,save_module,lr=0.01, lamb=0.1, epoch=100):                            
    
    rule_embeddings=rule_embeddings.cuda()
    # define the optimization

    consecutive_no_improvement = 0 


    optimizer = Adam(model.net.parameters(), lr=lr)
    # enumerate epochs
    for epoch in range(epoch):
        # enumerate mini batches
        current_train_loss = 0.
        for i_th_batch, (batch_x, batch_y) in enumerate(train_dl):

            pred, sub_result, sample_2_embeddings = model.forward(batch_x,batch_y)

            loss = model.criterion(batch_y, pred, sub_result)

            # calculate loss
            
            # Start OT Computation
            sample_2_embeddings = sample_2_embeddings.float()
            n_sample=batch_y.shape[0]
            n_rule=len(rule_embeddings)
            M_rule_and_data=ot.dist(sample_2_embeddings,rule_embeddings)
            a, b = torch.ones((n_sample,)) / n_sample, torch.ones((n_rule,)) / n_rule
            a=a.cuda()
            b=b.cuda()

            cost_matrix_minimum_figure = M_rule_and_data.max()  # avoiding Gradient explosion
            cost_matrix_minimum_figure_num=obtain_maximum_figure_num(cost_matrix_minimum_figure)
            reg = pow(10, cost_matrix_minimum_figure_num)
            if(n_sample > 100):
                Gs = ot.sinkhorn2(a, b, M_rule_and_data,reg=reg*1000)
            else:
                Gs = ot.sinkhorn2(a, b, M_rule_and_data,reg=reg*50)

            bce_loss_maximum_figure_num=obtain_maximum_figure_num(loss)
            ot_loss_maximum_figure_num=obtain_maximum_figure_num(Gs)

            if(ot_loss_maximum_figure_num<bce_loss_maximum_figure_num):
                difference_value=bce_loss_maximum_figure_num-ot_loss_maximum_figure_num
                for i in range(difference_value-1):
                    Gs=Gs*10
            elif(ot_loss_maximum_figure_num>bce_loss_maximum_figure_num):
                difference_value = ot_loss_maximum_figure_num-bce_loss_maximum_figure_num
                for i in range(difference_value-1):
                    Gs=Gs*0.1
             # End OT Computation and generate OT Loss based on OT distance
            loss+=lamb*Gs
            current_train_loss += loss.data
            # clear the gradients
            optimizer.zero_grad()
            # credit assignment
            loss.backward()
            # print("KDAlign_FeaWAD Training epoch: {}, batch: {}, loss: {}".format(epoch, i_th_batch, loss.data))
            # update model weights
            optimizer.step()
        
        print("epoch: {}, average_loss: {}".format(epoch, current_train_loss/(i_th_batch+1)))
        train_score, train_loss = None, None
        val_score, _,_ = evaluate_model_on_target_dataloader(val_dl, model) 

        #根据val_score的f1保存评估结果
        save_module.update_model_and_loss(current_train_loss=train_loss,current_model=model,current_train_score=train_score,current_val_score=val_score)                      
        if consecutive_no_improvement >= save_module.early_stop_num:
            print(f"Early stopping at epoch {epoch + 1} as there was no improvement in {save_module.early_stop_num} epochs.")
            break
        elif val_score['RECK'] > save_module.best_val_reck:
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
        test_score, _,_ = evaluate_model_on_target_dataloader(test_dl, model)     
        save_module.update_model_best_test_result(best_test_score=test_score)        
 
    return



if __name__=='__main__':
    #———————————————————————————————— 基于自动调参的训练和评估代码 ——————————————————————————————————————————————————————————————————
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed.', required=False)
    parser.add_argument('--dataset_name', type=str,default='Cardiotocography', required=False)
    parser.add_argument('--dataset_suffix_list', type=str,default=['_Complementary_Evaluation_Version'], nargs='+', required=False) #想传入一个不同划分版本的名称后缀列表
    parser.add_argument('--dataset_path',default='./data/', type=str, required=False)
    parser.add_argument('--records_saved_path', type=str, default='./Experimental_Results/Results/', required=False)
    parser.add_argument('--selected_device', type=int, default=0, help='number of selected gpu device')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.3, help='Control the GPU Memory Usage Per Process')
    parser.add_argument('--target_model', type=str, default='FeaWAD', help='Initialize Assigned Model')
    parser.add_argument('--repeat_num', type=int, default=1, help='The number of repeating experiments.') #同一个seed执行repeat_num次好了
    parser.add_argument('--labeled_anomaly_samples', type=int, default=10, help='The number of labeled anomaly samples in wsad.')   

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if(args.cuda):
        torch.cuda.set_device(args.selected_device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,args.selected_device) #控制每个进程的显卡使用率
        print('Using gpu ',torch.cuda.current_device())

    for arg in vars(args):
        print(f'{arg:>30s} = {str(getattr(args, arg)):<30s}')
    seed_name = '.seed' + str(args.seed)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False                

    records_saved_path = args.records_saved_path
    dataset_path = args.dataset_path 
    dataset_name = args.dataset_name 
    dataset_suffix_list = args.dataset_suffix_list
    target_model = args.target_model
    repeat_num = args.repeat_num
    labeled_anomaly_samples = args.labeled_anomaly_samples    

    for dataset_suffix in dataset_suffix_list:
        print(target_model)
        print('————————',dataset_name,'数据集在划分版本',dataset_suffix[1:],' WSAD下的FeaWAD和KDAlign_FeaWAD模型的调参过程开始————————\n')

        if(dataset_name=='Amazon'):
            epoch = [20,50]
            lr = [0.001, 0.01]
            hidden_dim=[32,64,128,256]
            layer_num = range(2, 4)
            lamb=[0.01,0.05,0.1,0.5,1.0,1.5,3.0]
        elif(dataset_name == 'Cardiotocography'):
            epoch = [20,100]
            lr = [0.001, 0.01]
            hidden_dim=[32,64,128,256]
            layer_num = range(2, 4)
            lamb=[0.01,0.05,0.1,0.5,1.0,1.5,3.0]

        FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics = {'epoch':[],'labeled_anomaly_samples':[],'layer_num':[],'learning_rate': [],'hidden_dim': [],'lamb':[],
                                                        'FeaWAD_Val_Best_F1_K': [], 'KDAlign_FeaWAD_Val_Best_F1_K': [],

                                                        'FeaWAD_Test_Best_AUCPR': [], 'KDAlign_FeaWAD_Test_Best_AUCPR': [],
                                                        'FeaWAD_Test_Best_F1_K': [], 'KDAlign_FeaWAD_Test_Best_F1_K': [],
                                                        'FeaWAD_Val_Avg_F1_K': [], 'KDAlign_FeaWAD_Val_Avg_F1_K': [],
                                                        'FeaWAD_Val_Std_F1_K': [], 'KDAlign_FeaWAD_Val_Std_F1_K': [],

                                                        'FeaWAD_Test_Avg_AUCPR': [], 'KDAlign_FeaWAD_Test_Avg_AUCPR': [],
                                                        'FeaWAD_Test_Std_AUCPR': [], 'KDAlign_FeaWAD_Test_Std_AUCPR': [],
                                                        'FeaWAD_Test_Avg_F1_K': [], 'KDAlign_FeaWAD_Test_Avg_F1_K': [],
                                                        'FeaWAD_Test_Std_F1_K': [], 'KDAlign_FeaWAD_Test_Std_F1_K': []}                         


        dataset_train_path = dataset_path+dataset_name+'/'+dataset_name+'_ML_Train'+dataset_suffix+'.csv'
        dataset_val_test_path= dataset_path+dataset_name+'/'+dataset_name+'_ML_Test_Evaluation_Version.csv'
        
        train_dataset = data_preprocess_wsad(dataset_path=dataset_train_path,labeled_anomaly_samples=labeled_anomaly_samples)
        val_test_dataset = data_preprocess_no_noise(dataset_path=dataset_val_test_path)
        attribute_names=train_dataset['attribute_names']

        train_samples=train_dataset['samples']
        train_labels=train_dataset['labels']

        val_test_samples=val_test_dataset['samples']
        val_test_labels=val_test_dataset['labels'] 
        val_test_propotion=[1,2]
        test_size=(val_test_propotion[1])/(sum(val_test_propotion))

        X_train, y_train = train_samples,train_labels
        X_val,X_test, y_val, y_test =train_test_split(val_test_samples,val_test_labels,test_size=test_size, random_state=args.seed)

        dataset_train = SampleDataset(X=X_train,y=y_train)
        dataset_val = SampleDataset(X=X_val,y=y_val)
        mean = 0
        add_noise = False        
        std_dev = 0.0
        dataset_test = SampleDataset(X=X_test,y=y_test,add_noise=add_noise, mean = mean, std_dev = std_dev)

        train_stage_dl = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
        val_stage_dl = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True) 
        test_stage_dl = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)

        print(dataset_train_path+"训练集(剔除规则样本)的样本数量：",len(train_stage_dl.dataset))           
        print(dataset_val_test_path+" 验证集集的样本数量：",len(val_stage_dl.dataset))  
        print(dataset_val_test_path+" 测试集的样本数量：",len(test_stage_dl.dataset))                               
        
        Rule_embedding_file= dataset_path+dataset_name+'/'+'formula_embedding_dict.pk'
        with open(Rule_embedding_file,'rb') as f:
            rule_embeddings=torch.stack(list(pk.load(f).values()))
        rule_dim = rule_embeddings[0].shape[0]  

        input_dim=len(train_stage_dl.dataset[0][0])

        for tuned_parameter_list in itertools.product(epoch,lr,hidden_dim,layer_num,lamb):
            epoch = tuned_parameter_list[0]
            lr=tuned_parameter_list[1]
            hidden_dim=tuned_parameter_list[2]
            layer_num=tuned_parameter_list[3]
            lamb=tuned_parameter_list[4]      
            
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False                
            
            FeaWAD_Val_F1_K_records=[]                
            FeaWAD_Test_AUCROC_records=[]
            FeaWAD_Test_AUCPR_records=[]
            FeaWAD_Test_F1_K_records=[]

            KDAlign_FeaWAD_Val_F1_K_records=[]
            KDAlign_FeaWAD_Test_AUCROC_records=[]
            KDAlign_FeaWAD_Test_AUCPR_records=[]
            KDAlign_FeaWAD_Test_F1_K_records=[] 

            for i_th_repeat in range(repeat_num):                   

                FeaWAD_model = FeaWAD(input_dim=input_dim,hidden_dims=(str(hidden_dim)+',')*(layer_num-1)+str(hidden_dim),hidden_dims2='256,'+str(rule_dim),random_state=args.seed)  # pure model
                KDAlign_FeaWAD_model = FeaWAD(input_dim=input_dim,hidden_dims=(str(hidden_dim)+',')*(layer_num-1)+str(hidden_dim),hidden_dims2='256,'+str(rule_dim),random_state=args.seed)  # 联合训练的Model

                FeaWAD_model.net.cuda()
                KDAlign_FeaWAD_model.net.cuda()        
                train_stage_dl = FeaWAD_model.wsad_training_prepare(dataset_train,batch_size=len(dataset_train))
                val_stage_dl = FeaWAD_model.validation_prepare(dataset_val.X,dataset_val.y,batch_size=len(dataset_val))
                test_stage_dl = FeaWAD_model.testing_prepare(dataset_test.X,dataset_test.y,batch_size=len(dataset_test))
                

                FeaWAD_model_record=Saved_Module_Based_On_Val(FeaWAD_model, early_stop_num=1000)
                KDAlign_FeaWAD_model_record=Saved_Module_Based_On_Val(KDAlign_FeaWAD_model, early_stop_num=1000)

                train_model(train_stage_dl,val_stage_dl, test_stage_dl,FeaWAD_model,save_module=FeaWAD_model_record,lr=lr,epoch=epoch) 
                KDAlign_FeaWAD_train_model(train_stage_dl,val_stage_dl, test_stage_dl,KDAlign_FeaWAD_model, rule_embeddings, save_module=KDAlign_FeaWAD_model_record,lr=lr,lamb=lamb, epoch=epoch)

                FeaWAD_Val_F1_K_records.append(FeaWAD_model_record.best_val_score['RECK'])
                KDAlign_FeaWAD_Val_F1_K_records.append(KDAlign_FeaWAD_model_record.best_val_score['RECK'])         

                FeaWAD_Test_AUCPR_records.append(FeaWAD_model_record.best_test_score['AUCPR'])                  
                KDAlign_FeaWAD_Test_AUCPR_records.append(KDAlign_FeaWAD_model_record.best_test_score['AUCPR'])                  

                FeaWAD_Test_F1_K_records.append(FeaWAD_model_record.best_test_score['RECK'])                    
                KDAlign_FeaWAD_Test_F1_K_records.append(KDAlign_FeaWAD_model_record.best_test_score['RECK'])                    

            # Save experimental results as a csv file

            saved_FeaWAD_and_KDAlign_FeaWAD_auto_tune_path=records_saved_path+dataset_suffix[1:]+'_WSAD_With_Validation/'+target_model+'/'+dataset_name+'_Best_and_Avg_FeaWAD_And_KDAlign_FeaWAD_Model_Auto_Tuned_Records'+dataset_suffix+'.csv'

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['epoch'].append(epoch)     
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['labeled_anomaly_samples'].append(labeled_anomaly_samples)   
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['learning_rate'].append(lr)
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['layer_num'].append(layer_num)
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['hidden_dim'].append(hidden_dim)
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['lamb'].append(lamb)


            best_FeaWAD_Val_F1_K = max(FeaWAD_Val_F1_K_records)
            best_FeaWAD_index = FeaWAD_Val_F1_K_records.index(best_FeaWAD_Val_F1_K)
            best_KDAlign_FeaWAD_Val_F1_K = max(KDAlign_FeaWAD_Val_F1_K_records)
            best_KDAlign_FeaWAD_index = KDAlign_FeaWAD_Val_F1_K_records.index(best_KDAlign_FeaWAD_Val_F1_K)

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Val_Best_F1_K'].append(FeaWAD_Val_F1_K_records[best_FeaWAD_index])
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Val_Best_F1_K'].append(KDAlign_FeaWAD_Val_F1_K_records[best_KDAlign_FeaWAD_index])

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Best_AUCPR'].append(FeaWAD_Test_AUCPR_records[best_FeaWAD_index])
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Best_AUCPR'].append(KDAlign_FeaWAD_Test_AUCPR_records[best_KDAlign_FeaWAD_index])

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Best_F1_K'].append(FeaWAD_Test_F1_K_records[best_FeaWAD_index])
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Best_F1_K'].append(KDAlign_FeaWAD_Test_F1_K_records[best_KDAlign_FeaWAD_index])

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Val_Avg_F1_K'].append(np.average(FeaWAD_Val_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Val_Avg_F1_K'].append(np.average(KDAlign_FeaWAD_Val_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Val_Std_F1_K'].append(np.std(FeaWAD_Val_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Val_Std_F1_K'].append(np.std(KDAlign_FeaWAD_Val_F1_K_records))


            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Avg_AUCPR'].append(np.average(FeaWAD_Test_AUCPR_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Avg_AUCPR'].append(np.average(KDAlign_FeaWAD_Test_AUCPR_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Std_AUCPR'].append(np.std(FeaWAD_Test_AUCPR_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Std_AUCPR'].append(np.std(KDAlign_FeaWAD_Test_AUCPR_records))

            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Avg_F1_K'].append(np.average(FeaWAD_Test_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Avg_F1_K'].append(np.average(KDAlign_FeaWAD_Test_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['FeaWAD_Test_Std_F1_K'].append(np.std(FeaWAD_Test_F1_K_records))
            FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics['KDAlign_FeaWAD_Test_Std_F1_K'].append(np.std(KDAlign_FeaWAD_Test_F1_K_records))              
            
            print(FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics)
            FeaWAD_And_KDAlign_FeaWAD_Model_statistical_table = pd.DataFrame.from_dict(FeaWAD_and_KDAlign_FeaWAD_model_para_and_acc_statistics)
            # FeaWAD_And_KDAlign_FeaWAD_Model_statistical_table.to_csv(saved_FeaWAD_and_KDAlign_FeaWAD_auto_tune_path, index=False)
            print("A FeaWAD and A KDAlign_FeaWAD Model are Saved!",'(epoch=',epoch,'labeled_anomaly_samples=',labeled_anomaly_samples,'learning_rate=',lr,'layer_num=',layer_num,'hideen_dim=',hidden_dim,'lamb=',lamb, 'FeaWAD_Val_Best_F1_K=',max(FeaWAD_Val_F1_K_records), 'FeaWAD_Val_Best_F1_K=', max(KDAlign_FeaWAD_Val_F1_K_records), 'FeaWAD_Test_Best_F1_K=',max(FeaWAD_Test_F1_K_records), 'KDAlign_FeaWAD_Test_Best_F1_K=', max(KDAlign_FeaWAD_Test_F1_K_records), ')') 