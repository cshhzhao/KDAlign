# Knowledge Representation
## Module 1 - Represent decision path as CNF formulae.
1. Execute `object_and_predicate_definition_1.py `
   - Define and extract objects and predicates utilized in the following Logical Formulae.

2. Execute `get_each_incident_annotations_2.py`
   - Generate annotation for each acquired decision path.

3. Execute  `find_rels.py `
   - Extract all the covered anomaly scenarios defined by rules based on decision paths.

4. Execute `tokenize_vocabs.py`
   - Generate NLP Embeddings for annotation, such as 'Anomaly', 'is', or other feature names.

5. Execute `rel2cnf.py`
   - Generate CNF formulae, and for each formula, including corresponding five true assignment formulae and five false formulae for subsequent contrastive learning to embed knowledge into a knowledge space.

6. Execute `cnf2ddnnf_6_linux.py` or `cnf2ddnnf_6_win.py`
   - Change the CNF format to D-DNNF format.

7. Execute `renameNNF_7.py`
   - Rename the files generate by *step 6*, which include *.cnf*.

8. Execute `relddnnf2data_8.py`
   - Obtain graph structure of d-dnnf formulae.

9. Execute `rule_and_data_driven_dataset_split_9_complementary_version.py`.
    - Delete anomaly samples that match acquired rules. Total deleted number is #Rule-detect.
    - Generate preprocessed training dataset, such as `Amazon_ML_Train_Complementary_Evaluation_Version.csv`.

10. Key files used for knowledge     representaion are in the folder `vrd_ddnnf`.

###  Tips for executing in Linux and Windows Operation System**
1. if you run the code in a linux system, you should select `cnf2ddnnf_6_linux.py`
2. if you run the code in a windows system, you should select `cnf2ddnnf_6_win.py`.
3. In a linux system, if you use our provided **c2d** program, please execute `chmod +x c2d_linux`

##  Module 2 - Training Knowledge Encoder
1. `cd ./tree_rule_process/model/pygcn/pygcn`
2. GPU Version
   - `python train_with_gpu_no_val.py --ds_path ../../../Amazon --dataset vrd_ddnnf --epochs 2000 --hidden 64 --dataloader_worker 0 --w_reg 0.1 --indep_weight --selected_device 7`
3. CPU Version
   - `python train_only_cpu_no_val.py --ds_path ../../../Amazon --dataset vrd_ddnnf --epochs 2000 --hidden 64 --dataloader_worker 0 --no-cuda  --w_reg 0.1 --indep_weight`
4. Bash Version **(Recommended)**
   - `bash ./scripts/train_rule_embeddings.sh`
   - simultaneously training amazon and cardiotocography datasets.
##  Module 3 - Save Knowledge (Rule) Embeddings
1. Execute `tree_rule_process/model/pygcn/pygcn/obtain_each_formula_embedding_dict_gpu_version.py`
   - Compute knowledge embeddings and save them as a pickle file.
   
2. (Optional) you can also use `tree_rule_process/model/pygcn/pygcn/obtain_each_formula_embedding_dict_cpu_version.py`