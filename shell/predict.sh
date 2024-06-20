model_name_or_path="/mnt/nvme_share/common/LLMs/chinese-llama-2-7b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CHIP-CTC"  # 填入数据集所在的文件夹路径
CHECKPOINT="/mnt/nvme_share/srt07/PromptCBLUE-main/experiments/CHIP-CTC/promptcblue-llama-7b-pt-v0"   # 填入用来存储模型的文件夹路径

STEP=12500    # 用来评估的模型checkpoint是训练了多少步

CUDA_VISIBLE_DEVICES="1" python generate/predict.py \
    --do_predict \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/test.json \
    --cache_dir $your_data_path/cache \
    --overwrite_cache \
    --prompt_column input   \
    --response_column target    \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --peft_path $CHECKPOINT/checkpoint-$STEP \
    --output_dir $your_data_path/trainB2 \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 200 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --max_source_length 828 \
    --max_target_length 196 \
    --predict_with_generate
