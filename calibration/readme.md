# 本文件夹为校准器训练与评估

requirements.txt        校准器模型环境配置

LLaMa_generate.py       获取大模型内部状态，可以获取不同层的embedding和注意力分数并保存
TrainProbes.py          通过上一步获取的内部状态，进行校准器训练和预测
Generate_predictions.py 利用训练好的校准器模型，进行预测（事实上这一步一般用不到，上一步同时包含了预测）

以上三个文件的参数包含在config.json内，通过修改config.json，可以直接python LLaMa_generate.py以运行
以下对config.json的参数进行解释：
    "model_path" : 大模型文件路径，是已经合并微调参数后的大模型
    "task" : 校准器类型，仅有"cla"和"reg"两个选项。前者代表使用二分类器模型，数据集label必须是0或1；后者代表使用线性回归模型，label可以是浮点数
    "model": 模型类型，为"llama-7b"。也可以修改，这个参数用于文件名生成的标识，保证前后一致即可，对加载模型无影响
    "parameter": 参数类型，可选"embeddings"，"max_a"，"min_a"，"abs_a"，为csv文件中参数列名，可根据参数名修改
    "remove_period": 是否去除不必要符号，通常选false
    "test_first_only": 是否仅以list_of_datasets中第一个数据集作为测试集，选True。如果选false,会将每一个数据集作为测试集，剩下的作为训练集评估
    "save_probes": 是否将训练好的校准器模型保存，用于Generate_predictions.py  
    "repeat_each": 重复训练次数以获得最佳模型
    "probes_dir": 模型保存地址，配合save_probes参数
    "layers_to_use": 获取哪些隐藏层的参数以训练
    "layer": 调用哪一层参数训练好的模型，用于Generate_predictions.py
    "list_of_datasets": 数据集列表
    "dataset_path": 包含statement和label的原始数据集位置
    "processed_dataset_path": 生成参数后的保存位置
    "true_false": 后缀是否添加"true_false"，无影响
    "batch_size": 
    "gen_predictions_dataset": 文件名缀词，用于Generate_predictions.py
    "gen_predictions_layer": 文件名用词，用于Generate_predictions.py（这两个参数用于配合选择文件）
    "suffix_list": 后缀列表，本任务用不上

calibration.py      绘制校准图,计算ECE和MCE指标;可通过/shell同名脚本调用