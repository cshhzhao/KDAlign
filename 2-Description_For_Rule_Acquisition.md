# Descriptions For Folder `Rules_based_on_Decision_Trees`

## File descriptions
1. Rules_based_on_Decision_Trees/DecisionTrees.py 文件
   - Training several decision tree models.
2. DecisionTree_auto_optimization.py
  - It is used to train multiple Decision Tree models at the same time, and each Decision Trees model has a corresponding ID number
  - The core of this file is automatic parameter tuning, which finds multiple model parameters at the same time through multi-threaded operations and outputs the model results corresponding to each parameter(全部正确的路径数量和对应的样本数量)
  - In order to better find the rules, a further restriction is set here; that is, when all the correct path traversals are completed, only the exact correct path information corresponding to the sample size greater than 5 is retained.
  - Debug on IDE, not directly on a bash file. Note that selected_path_sample_num and specific_saved_num are two key variables.
  - Note that in each DecisionTrees model, it is more appropriate to set the `max_features` parameter to `auto`, and it is easy to get a better division of paths.

----
## Saved Decision Tree Model Descriptions
1. ### Amazon Dataset
   - **Amazon_Decision_Tree_List.pkl**
    - `max_depth= 10`, `max_features= 15` For each trained DT model.
    - Total five trained decision tree models
    - **Dataset description in Manuscript:**
      - **Total samples:** 11944
      - **Features**: 25
      - **Rule:** 20
      - **#Rule-Detect:** 431
      - **Label:** 821
      - **Rate:** 52.0%

2. ### Cardiotocography
   - **Cardiotocography_Decision_Tree_List.pkl**
    - `max_depth= 10`, `max_features= 9` For each trained DT model.
    - Total five trained decision tree models
    - **Dataset description in Manuscript:**
      - **Total samples:** 2144
      - **Features**: 21
      - **Rule:** 14
      - **#Rule-Detect:** 281
      - **Label:** 466
      - **Rate:** 60.0%