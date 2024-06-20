import json
import random
import pandas as pd
import numpy as np
import difflib
import csv
import argparse


def format(data, task_type:str, is_predict:int):
    
    formatted_data_list = []

    if task_type == 'CMeEE-V2':

        # 随机选择input模板
        input_templates = [
            "找出指定的实体：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]\n答：",
            "找出指定的实体：\n[INPUT_TEXT]\n实体类型选项：[LIST_LABELS]\n答：",
            "找出句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
            "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]实体是什么？\n答：",
            "生成句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
            "下面句子中的[LIST_LABELS]实体有哪些？\n[INPUT_TEXT]\n答：",
            "实体抽取：\n[INPUT_TEXT]\n选项：[LIST_LABELS]\n答：",
            "医学实体识别：\n[INPUT_TEXT]\n实体选项：[LIST_LABELS]\n答："
        ]

        # 新的映射字典
        # 九大类，包括：疾病(dis)，临床表现(sym)，药物(dru)，医疗设备(equ)，医疗程序(pro)，身体(bod)，医学检验项目(ite)，微生物类(mic)，科室(dep)
        new_category_mapping = {
            "dis":"疾病",
            "ite":"医学检验项目",
            "dep":"医院科室",
            "bod":"身体部位",
            "mic":"微生物类",
            "sym":"临床表现",
            "dru":"药物",
            "equ":"医疗设备",
            "pro":"医疗程序"
        }

        keyword = list(new_category_mapping.keys())

        # 存储所有格式化数据为一个文件，每行一个 JSON 对象

        index=0
        for item in data:
            input_template = random.choice(input_templates)
            entities_list = []
            dict_sample={}
            for k in np.random.choice(keyword,8,replace=False):
                v = new_category_mapping[k]
                dict_sample.update(dict({k:v}))
                entities_list.append(new_category_mapping[k]+"实体：")
                for entity in item["entities"]:
                    if entity["type"] == k:
                        entities_list.append(entity["entity"])
                        entities_list.append(',')
                if entities_list[-1] == ',':
                    entities_list.pop()
                entities_list.append('\n')
            index=index+1
            formatted_item = {
                "input": input_template.replace("[INPUT_TEXT]", item["text"]).replace("[LIST_LABELS]", ",".join(dict_sample.values())),
                "target": "" if is_predict else "上述句子中的实体包含：\n"+''.join(entities_list),
                "answer_choices": list(dict_sample.values()),
                "task_type": "ner",
                "task_dataset": "CMeEE-V2",
                "sample_id": index
            }  
            formatted_data_list.append(formatted_item)

    elif task_type == 'CHIP-CTC':

        input_templates = [
            "判断临床试验筛选标准的类型：\n[INPUT_TEXT]\n选项：[LIST_LABELS]\n答：",
            "确定试验筛选标准的类型：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]\n答：",
            "[INPUT_TEXT]\n这句话是什么临床试验筛选标准类型？\n类型选项：[LIST_LABELS]\n答：",
            "[INPUT_TEXT]\n是什么临床试验筛选标准类型？\n选项：[LIST_LABELS]\n答：",
            "请问是什么类型？\n[INPUT_TEXT]\n临床试验筛选标准选项：[LIST_LABELS]\n答："
        ]

        new_category_mapping = {
            "Disease": "疾病",
            "Symptom": "症状(患者感受)",
            "Sign": "体征(医生检测）",
            "Pregnancy-related Activity": "怀孕相关",
            "Neoplasm Status": "肿瘤进展",
            "Non-Neoplasm Disease Stage": "疾病分期",
            "Allergy Intolerance": "过敏耐受",
            "Organ or Tissue Status": "器官组织状态",
            "Life Expectancy": "预期寿命",
            "Oral related": "口腔相关",
            "Pharmaceutical Substance or Drug": "药物",
            "Therapy or Surgery": "治疗或手术",
            "Device": "设备",
            "Nursing": "护理",
            "Diagnostic": "诊断",
            "Laboratory Examinations": "实验室检查",
            "Risk Assessment": "风险评估",
            "Receptor Status": "受体状态",
            "Age": "年龄",
            "Special Patient Characteristic": "特殊病人特征",
            "Literacy": "读写能力",
            "Gender": "性别",
            "Education": "教育情况",
            "Address": "居住情况",
            "Ethnicity": "种族",
            "Consent": "知情同意",
            "Enrollment in other studies": "参与其它试验",
            "Researcher Decision": "研究者决定",
            "Capacity": "能力",
            "Ethical Audit": "伦理审查",
            "Compliance with Protocol": "依存性",
            "Addictive Behavior": "成瘾行为",
            "Bedtime": "睡眠",
            "Exercise": "锻炼",
            "Diet": "饮食",
            "Alcohol Consumer": "酒精使用",
            "Sexual related": "性取向",
            "Smoking Status": "吸烟状况",
            "Blood Donation": "献血",
            "Encounter": "病例来源",
            "Disabilities": "残疾群体",
            "Healthy": "健康群体",
            "Data Accessible": "数据可及性",
            "Multiple": "含有多类别的语句"
        }


        for item in data:
            input_template = random.choice(input_templates)
            formatted_item = {
                "input": input_template.replace("[INPUT_TEXT]", item["text"]).replace("[LIST_LABELS]", ",".join(new_category_mapping.values())),
                "target": "" if is_predict else new_category_mapping[item["label"]],
                "answer_choices": list(new_category_mapping.values()),
                "task_type": "cls",
                "task_dataset": "CHIP-CTC",
                "sample_id": item["id"]
            }
            formatted_data_list.append(formatted_item)

    elif task_type == 'CHIP-STS':

        input_templates = [
            "以下两句话的意思相同的吗？\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”\\n选项：是的，不是\\n答：",
            "我想知道下面两句话的意思是否相同。\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n选项：是的，不是\\n答：",
            "我是否可以用以下的句子：“[INPUT_TEXT_1]”，来替换这个句子：“[INPUT_TEXT_2]”，并且它们有相同的意思？\\n选项：是的，不是\\n答：",
            "“[INPUT_TEXT_1]”和“[INPUT_TEXT_2]”是同一个意思吗？\\n选项：是的，不是\\n答：",
            "“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n这两句是一样的意思吗？\\n选项：是的，不是\\n答："
        ]

        new_category_mapping = {
            "1":"是的",
            "0":"不是"
        }

        for item in data:
            input_template = random.choice(input_templates)
            formatted_item = {
                "input": input_template.replace("[INPUT_TEXT_1]", item["text1"]).replace("[INPUT_TEXT_2]", item["text2"]),
                "target":"" if is_predict else new_category_mapping[item["label"]],
                "answer_choices": ["相同", "不同"],
                "task_type": "cls",
                "task_dataset": "CHIP-STS",
                "sample_id": item["id"]
            }
            formatted_data_list.append(formatted_item)

    return formatted_data_list



def main():
    parser = argparse.ArgumentParser(description='Generate Formatted Dataset.')

    parser.add_argument('--input_file', help='pre-data CBLUE')
    parser.add_argument('--output_file', help='out_data with prompt')
    parser.add_argument('--is_predict', type=int, help='if generate blank target or not')
    parser.add_argument('--task_type')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    is_predict = args.is_predict
    task_type = args.task_type

    # 读取JSON文件
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    formatted_data_list = format(data,task_type,is_predict)

        # 将格式化后的数据以一行一个 JSON 对象的格式写入文件
    with open(output_file, 'w') as output_file:
        for formatted_item in formatted_data_list:
            json.dump(formatted_item, output_file, ensure_ascii=False)
            output_file.write('\n')  # 在 JSON 对象之间换行分隔

if __name__ == "__main__":
    main()