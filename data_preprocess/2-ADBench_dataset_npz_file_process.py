import numpy as np
from num2words import num2words as n2w
import pandas as pd
file_path="./data/Raw_Data/Cardiotocography.npz"
poem=np.load(file_path,allow_pickle=True)

save_info=[]
for key, arr in poem.items():
  save_info.append(arr)
samples=save_info[0]
labels=save_info[1]

samples_and_labels=np.concatenate((samples,labels.reshape((-1,1))),axis=1)
column_names=[]
for i in range(1,len(samples_and_labels[0])+1):
  column_names.append(n2w(i))
column_names[-1]='Outcome'
dataset=pd.DataFrame(samples_and_labels,columns=column_names)
dataset.to_csv('./data/Cardiotocography.csv',index=False)