lr=1e-4
lora_rank=8
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
#lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
#pretrained_model="./resources/chinese-llama-plus-lora-7b"
pretrained_model="/mnt/nvme_share/common/LLMs/chinese-llama-2-7b"
dataset_name="./datasets/CHIP-STS"
dataset_cache_dir="./datasets/CHIP-STS/cache"
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
training_steps=8000
# training_epochs = 5
output_dir="./experiments/CHIP-STS/promptcblue-llama-7b-pt-v0"
# deepspeed_config_file="src/chatmed_llama_peft/deepspeed_config_zero3_offload.json"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# CUDA_VISIBLE_DEVICES=0 

CUDA_VISIBLE_DEVICES=0,1
torchrun \
 --nnodes 1 \
 --nproc_per_node 2 \
 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12356 \
  generate/train_lora.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_name ${dataset_name} \
    --dataset_cache_dir ${dataset_cache_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 100 \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 400 \
    --save_strategy steps \
    --save_total_limit 25 \
    --save_steps 1600 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16

#   --deepspeed ${deepspeed_config_file} \

#   --num_train_epochs ${training_epochs}\
#   

#   --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12356 \