CUDA_VISIBLE_DEVICES="0" python generate/merge.py \
    --base_model /mnt/nvme_share/common/LLMs/chinese-llama-2-7b/ \
    --lora_model /mnt/nvme_share/srt07/PromptCBLUE-main/experiments/CHIP-STS/promptcblue-llama-7b-pt-v0/checkpoint-8000/ \
    --output_type huggingface \
    --output_dir /mnt/nvme_share/srt07/PromptCBLUE-main/experiments/CHIP-STS/promptcblue-llama-7b-pt-v0/checkpoint-8000-merge/