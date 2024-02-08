# Description For Data Preprocess
## Folde Name：`data_preprocess`
- **1-graph_Dataset_Proprecess.py**\
  For preprocessing Amazon Dataset\
  Ignoring the graph structure\
  Based on DGL library.
- **2-ADBench_dataset_npz_file_process.py**\
  For preprocessing .npz files provided by NIPS 2022 ADBench
- **3-split_train_val_and_test_in_advance.py**\
  Split training and test dataset by 7:3. The training set is further preprocessed by deleting the anomal samples that match corresponding rules. Note that Test dataset is furthere splitted into validation and test set by 1:2 during the evaluatoin stages.\
  *The final ratio of train, val, test is 7:1:2*
