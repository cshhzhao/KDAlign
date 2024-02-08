from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
import torch
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss,Softmax
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import pickle as pk
import ot
import ot.plot
import os
import numpy as np
import copy

class Saved_Module_Based_On_Val: 
    def __init__(self,initialized_model, early_stop_num=10):      
        self.model=initialized_model
        
        self.best_val_reck = -np.inf

        self.best_train_loss = 0.0
        
        self.best_train_score = {}          
        self.best_val_score = {}                
        self.best_test_score = {}   
        self.early_stop_num = early_stop_num 
        self.early_stop = False 
     
    def update_model_and_loss(self,current_train_loss,current_train_score,current_val_score,current_model):
        if(current_val_score['RECK']>=self.best_val_reck):
            self.model=copy.deepcopy(current_model)
            
            self.best_val_reck=current_val_score['RECK']

            self.best_train_loss=current_train_loss
            
            self.best_val_score=current_val_score                          
            self.best_train_score=current_train_score              
        # else:
        #     if self.early_stop_num > 0:
        #         self.early_stop_num -= 1  # 每次验证不进步时减1
        #         if self.early_stop_num == 0:
        #             self.early_stop = True  # 达到早停条件

            return

    def update_model_best_test_result(self,best_test_score):
        self.best_test_score=best_test_score                

        return