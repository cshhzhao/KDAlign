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

save_path = './Breast_Cancer_Wisconsin/tuple_right_path_and_sample.pk'
tuple_right_path_and_sample=pk.load(open(save_path, 'rb'))
decision_path_sample_num=0
for tree_index,path_key_sample_list in enumerate(tuple_right_path_and_sample[1]):
    for path_key,sample_list in path_key_sample_list.items():
        decision_path_sample_num+=len(sample_list)
print(decision_path_sample_num)

#602个样本