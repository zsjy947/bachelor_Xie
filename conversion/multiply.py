import json
import random
import pandas as pd
import numpy as np
import difflib
import csv
import argparse
import ast

def main():

    parser = argparse.ArgumentParser(description='Multiply')
    parser.add_argument('--layers', type=str, required=True,help='a list of integers representing layers')
    parser.add_argument('--datas', type=str)

    args = parser.parse_args()

    layers = ast.literal_eval(args.layers)
    datas = ast.literal_eval(args.datas)

    for layer in layers:
        for data in datas :
            path1 = f'/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/processed_data/embeddings_{data}llama-7b_{layer}.csv'
            path2 = f'/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/processed_data/attentions_{data}llama-7b_{layer}.csv'

            outfile = f'/mnt/nvme_share/srt07/Probes_Xie/CBLUEdatasets/CMeEE-V2/attention/embeddings_{data}llama-7b_{layer}.csv'
                
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)
            df = df2.copy()
            df2['str'] = df2['min_a'].str.replace(']','')+','+df2['max_a'].str.replace('[','')
            df['embeddings'] = df1['embeddings'].str.replace(']','')+','+df2['str'].str.replace('[','')
            df.to_csv(outfile,index=False,columns=['statement','label','parameters'])

if __name__ == "__main__":
    main()
