# define variable
port=18765
model_family="phi"
split="forget10"
#checkpoint_list=(10350) #Changed from 5553
#precheckpoint_list=(6624) #Already tried 33120, 6624, 19872, 13248, 26496
checkpoint_list=(5553)
precheckpoint_list=(6210)
minus_values=(5.0)


for i in "${!checkpoint_list[@]}"; do
    checkpoint=${checkpoint_list[$i]}
    precheckpoint=${precheckpoint_list[$i]}
    
    for minus in "${minus_values[@]}"; do
        echo "Processing checkpoint: $checkpoint, precheckpoint: $precheckpoint, minus: $minus"
        model_path="/projects/unlearning_models/final_ft_noLORA_5_epochs_inst_lr1e-05_${model_family}_full_minus_${split}_seed42_1/checkpoint-${checkpoint}"
        pretrained_path="/projects/unlearning_models/final_ft_noLORA_5_epochs_inst_lr1e-05_${model_family}_full_seed42_1/checkpoint-${precheckpoint}"

        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py \
            model_family=$model_family \
            batch_size=20 \
            split=$split \
            model_path=$model_path \
            +minus_value=$minus \
            +pretrained_path=$pretrained_path \
            --config-name=eval_idea.yaml
    done
done
#Batch size was 100, but ran into storage errors so I reduced it to 10 (did the same for training)
#Updated to 20 (system was able to handle it perfectly fine)
