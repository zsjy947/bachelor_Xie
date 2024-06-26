pre_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CMeEE-V2/test/test_prediction.json"
dev_file="/mnt/nvme_share/srt07/PromptCBLUE-main/datasets/CMeEE-V2/test/test_dev.json"
out_file="/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/pre-data/test.csv"    

task_type="CMeEE-V2"
label_file="/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/pre-data/test_label.csv"

python conversion/label.py \
    --pre_file      $pre_file \
    --dev_file      $dev_file \
    --out_file      $out_file \
    --task_type     $task_type \
    --label_file    $label_file