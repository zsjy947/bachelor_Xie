
## 验证：LLM回复转化为结构化格式的代码
python post_generate_process.py dev_predictions.json results.json

## 评分，input_param.json包含作为验证的原始数据和生成的测试数据
python evaluate.py input_param.json eval_result.json
cat eval_result.json


