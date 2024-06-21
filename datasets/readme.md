# 本文件夹包含三个任务的基础数据集

三个数据集文件结构相同，均包含
trainA.json
/trainB
/test

分别对应训练集A，训练集B，测试集T

其中trainA.json为原始数据，用于微调大模型

/trainB和/tests结构相同，包含test、dev和prediction三个文件

test文件target为空，等待大模型生成；dev为原始数据；prediction为大模型生成数据

现有数据集已经完成了大模型微调+数据集生成；但是若提取后续内部状态，仍需要微调训练的大模型。