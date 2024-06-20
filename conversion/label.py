import json
import random
import csv
import pandas as pd
import argparse

def main():

    parser = argparse.ArgumentParser(description='Generate labels.')

    parser.add_argument('--pre_file', help='pre_data with prediction')
    parser.add_argument('--dev_file', help='pre_data')
    parser.add_argument('--out_file', help='out_data with label')
    parser.add_argument('--task_type')
    parser.add_argument('--label_file',help='label for task CMeEE-V2')

    args = parser.parse_args()

    pre_file = args.pre_file
    dev_file = args.dev_file
    out_file = args.out_file
    task_type = args.task_type
    label_file = args.label_file

    input_templates = ["[INPUT][TARGET]"]
    pre_data=[]
    dev_data=[]
    data = []

    with open(pre_file, 'r') as file1:
        for line in file1.readlines():
            data = json.loads(line)
            pre_data.append(data) 

    if task_type == 'CMeEE-V2':

        df = pd.read_csv(label_file,'r')
        for index,row in df.iterrows():
            dev_data.append(row['label'])
        
        # 存储所有格式化数据为一个文件，每行一个 JSON 对象

        for (item1,item2) in zip(pre_data,dev_data):
            input_template = random.choice(input_templates)
            dict = {
                "statement": input_template.replace("[INPUT]", item1["input"]).replace("[TARGET]",item1["target"]),
                "label":item2,
                "logits" : item1['probs']
            }
            data.append(dict)

    else:

        with open(dev_file, 'r') as file2:
            for line in file2.readlines():
                data = json.loads(line)
                dev_data.append(data)
                
        for (item1,item2) in zip(pre_data,dev_data):
            input_template = random.choice(input_templates)
            dict = {
                "statement": input_template.replace("[INPUT]", item1["input"]).replace("[TARGET]",item1["target"]),
                "label": 1 if item1['target'] == item2['target'] else 0,
                "logits" : item1['probs']
            }
            data.append(dict)

    keys = data[0].keys()

    
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    main()


