master_port=18761
model=phi
lr=7e-6 #Halving the learning rate to check for improvements
batch_size=2 #Changed from 4
gradient_accumulation_steps=4

splits=("full")


for split in "${splits[@]}"; do
    echo "Processing split: $split"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=$master_port \
    finetune_v2.py --config-name=finetune_v2.yaml split=${split} \
    batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} \
    model_family=${model} lr=${lr} 
done



