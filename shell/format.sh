input_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CHIP-CTC/test/test3.json"     # 输入文件位置
output_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CHIP-CTC/test/pre3.json"     # 输出文件位置
is_predict=1  #为1则将target挖空，以待后续生成；为0则将原始的target加上
task_type="CHIP-CTC"


python conversion/format.py \
    --input_file    $input_file \
    --output_file   $output_file \
    --is_predict    $is_predict \
    --task_type     $task_type