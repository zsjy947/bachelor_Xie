# 本文件夹为PromptCBLUE数据集及评估文件

requirements.txt    PromptCBLUE需要的环境配置，与之后的校准器环境配置有冲突

train_lora.py       对大模型进行lora微调，生成lora参数并保存

merge.py            将lora参数与原始大模型合并获取微调后的大模型

predict.py          利用trainer.predict()对数据集进行生成，速度较快，但是生成目前有些问题

logits.py           利用model.generate()对数据集进行生成，并获取作为baseline的logits，速度很慢

注：其中merge.py可以不运行。运行后会保存一个完整的模型，占据很大的空间。
对于lora微调后的模型调用有两种方法，一种就是merge后调用新的模型；另一种是调用基础模型后，通过peft库调用保存的微调参数添加到基础模型上
两者没有本质区别，predict.py用的是后者，logits.py则用的是前者，因此脚本修改时注意model_name_or_path参数

以上文件均可以通过/shell同名脚本调用