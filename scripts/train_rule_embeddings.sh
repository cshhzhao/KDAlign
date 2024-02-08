cd ../tree_rule_process/model/pygcn/pygcn/

dataset_name="--dataset vrd_ddnnf"

ind_options="--indep_weight"
reg_options="--w_reg 0.1"
dataloader_worker="--dataloader_worker 0"
hidden_dim="--hidden 128"

#gpu version

selected_device="--selected_device 0"
target_dataset_name="--target_dataset_name Amazon"
gpu_memory_fraction="--gpu_memory_fraction 0.1"
nohup python train_with_gpu_no_val.py --ds_path ../../../Amazon ${dataset_name} --epochs 1000 ${hidden_dim} ${dataloader_worker} ${reg_options} ${ind_options} ${selected_device} ${target_dataset_name} ${gpu_memory_fraction} &>./model_save/Amazon/log.out & echo $! > ./model_save/Amazon/PID_FILE_NAME

selected_device="--selected_device 0"
target_dataset_name="--target_dataset_name Cardiotocography"
gpu_memory_fraction="--gpu_memory_fraction 0.1"
nohup python train_with_gpu_no_val.py --ds_path ../../../Cardiotocography ${dataset_name} --epochs 1000 ${hidden_dim} ${dataloader_worker} ${reg_options} ${ind_options} ${selected_device} ${target_dataset_name} ${gpu_memory_fraction} &>./model_save/Cardiotocography/log.out & echo $! > ./model_save/Cardiotocography/PID_FILE_NAME