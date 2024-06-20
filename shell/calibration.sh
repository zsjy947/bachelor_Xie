task_list = "['CHIP-STS','CHIP-CTC','CMeEE-V2']"
prob_list = "['logits','prob_1','prob_2','prob_3','prob_4','prob_5','prob_6','prob_7','prob_8']"
title = "['logits','layer=-1','layer=-2','layer=-3','layer=-4','layer=-5','layer=-6','layer=-7','layer=-8']"

python calibration/calibration.py \
    --prob_list  $prob_list \
    --title    $title\
    --task_list $task_list
