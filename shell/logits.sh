model_name_or_path="/mnt/nvme_share/srt07/PromptCBLUE-main/experiments/CHIP-CTC/promptcblue-llama-7b-pt-v0/checkpoint-12500-merge"   # LLM模型路径
input_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CHIP-CTC/test/test3.json"     # 输入文件位置
output_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CHIP-CTC/test/pre3.json"     # 输出文件位置
max_new_tokens=30 #根据任务输出长度调整，STS/CTC较短，CMeEE较长可以调到160；影响生成速度

CUDA_VISIBLE_DEVICES="1" python generate/logits.py \
    --model_name_or_path    $model_name_or_path \
    --input_file    $input_file \
    --output_file   $output_file \
    --max_new_tokens    $max_new_tokens