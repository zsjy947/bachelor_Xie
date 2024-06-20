layers = "[1,2,3,4,5,6,7,8]"
new_names = "['prob_1','prob_2', 'prob_3','prob_4','prob_5','prob_6','prob_7', 'prob_8']"
task_list = "['CMeEE-V2','CHIP-STS','CHIP-CTC']"

python conversion/merge_data.py \
    --layers $layers \
    --new_names $new_names \
    --task_list $task_list