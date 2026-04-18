#Command for running this script: bash finetune_phi_all_iter_v2.sh
master_port=18761
model=phi
lr=1e-5
batch_size=4 #Batch size can be lowered if memory issues are encountered.
gradient_accumulation_steps=4

splits=("full" "full_minus_forget10")
#Note: Torch needs to be installed within conda env for running this script

for split in "${splits[@]}"; do
    echo "Processing split: $split"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=$master_port \
    finetune_v2.py --config-name=finetune_v2.yaml split=${split} \
    batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} \
    model_family=${model} lr=${lr} 
done



