cd ../

dataset_path="--dataset_path ./data/"
dataset_suffix_list="--dataset_suffix _Complementary_Evaluation_Version" 

records_saved_path="--records_saved_path ./Experimental_Results/Results/"


target_model='--target_model FeaWAD'
seed="--seed 1024"
repeat_num="--repeat_num 1"

labeled_anomaly_samples="--labeled_anomaly_samples 10"

# gpu
dataset_name="--dataset_name Amazon"
selected_device="--selected_device 1"
gpu_memory_fraction="--gpu_memory_fraction 0.2" 
nohup python evaluation_auto_tuned_WSAD_KDAlign_Best_Performance.py ${dataset_path} ${dataset_name} ${target_model} ${dataset_suffix_list} ${records_saved_path} ${seed} ${selected_device} ${repeat_num} ${labeled_anomaly_samples} ${gpu_memory_fraction} &>./Experimental_Results/Log/Log_with_val/FeaWAD/log_Best_and_Avg_Few_Shot_amazon_10_shot.out & echo $! > ./Experimental_Results/Log/Log_with_val/FeaWAD/Amazon_Best_and_Avg_Few_Shot_10_shot_PID_FILE_NAME

dataset_name="--dataset_name Cardiotocography"
selected_device="--selected_device 1"
gpu_memory_fraction="--gpu_memory_fraction 0.2" 
nohup python evaluation_auto_tuned_WSAD_KDAlign_Best_Performance.py ${dataset_path} ${dataset_name} ${target_model} ${dataset_suffix_list} ${records_saved_path} ${seed} ${selected_device} ${repeat_num} ${labeled_anomaly_samples} ${gpu_memory_fraction} &>./Experimental_Results/Log/Log_with_val/FeaWAD/log_Best_and_Avg_Few_Shot_Cardiotocography_10_shot.out & echo $! > ./Experimental_Results/Log/Log_with_val/FeaWAD/Cardiotocography_Best_and_Avg_Few_Shot_10_shot_PID_FILE_NAME

