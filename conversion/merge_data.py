import json
import random
import pandas as pd
import numpy as np
import difflib
import csv

import argparse
import ast

def main():

    parser = argparse.ArgumentParser(description='Merge Data')
    parser.add_argument('--task_list', type=str, required=True)
    parser.add_argument('--new_names', type=str)
    parser.add_argument('--layers', type=str)

    args = parser.parse_args()

    task_list = ast.literal_eval(args.task_list)
    new_names = ast.literal_eval(args.prob_list)
    layers = ast.literal_eval(args.title)

    for task in task_list:

        csv_files = []

        test = f"/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/{task}/pre-data/test.csv"
        outfile = f"/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/{task}/pre-data/merged_data.csv"

        # 读取前四个csv文件并提取average_probability列
        for layer in layers:
            path = f"/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/{task}/processed_data/embeddings_testllama-7b_{layer}_reg_predictions.csv"
            csv_files.append(path)
        dataframes = [pd.read_csv(file,usecols=['average_probability']).rename(columns={'average_probability':new_names[i]}) for i,file in enumerate(csv_files)]

        # 读取第五个csv文件并提取test和logits列
        test_df = pd.read_csv(test)[['label'],['logits']]

        # 将所有数据合并到一个DataFrame中
        merged_df = pd.concat([test_df]+ dataframes , axis=1)
        merged_df.to_csv(outfile, index=False)

if __name__ == "__main__":
    main()
