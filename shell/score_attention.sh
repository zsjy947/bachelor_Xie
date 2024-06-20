data_dir = "/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/processed_data"    #需要处理的各层attention文件存在的位置
layers_to_use = "[1,2,3,4,5,6,7,8]" #选择了哪些层生成注意力分数
task_list = "['test','trainB']"

python conversion/score_attention.py \
    --data_dir  $data_dir \
    --layers    $layers_to_use\
    --task_list $task_list
