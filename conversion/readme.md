# 本文件夹为数据转换文件

format.py       将CBLUE的结构化转化为适用于PromptCBLUE的基于prompt的数据集

/evaluation     对生成的数据集进行评估，采用的是PromptCBLUE的评测基准
                这一步需要两个文件，一个是原始的转化为prompt的数据，另一个是生成target的数据；将这两个文件进行对比后评估
                特别的，对CMeEE-V2任务，会生成一个label.csv的文件，包含每条数据的标签，均为浮点数，表示该条数据的正确率。

label.py        对生成的数据集进行标签生成。其中分类任务STS/CTC只需简单对比即可生成0或1的label，CMeEE-V2则通过评估生成的label.csv写入label
                这一步生成的是csv文件，仅包含statement和label两列

score_attention.py  将生成的注意力分数提取出max_a、min_a和abs_a特征向量，并保存到新文件中

merge_data.py       将不同层生成的预测值合并，用于后续的校准图生成和指标计算对比

multiply.py         多参数合并校准

以上均可以通过/shell同名脚本运行