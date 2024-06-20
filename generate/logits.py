import sys

sys.path.append('./')
import os

import datasets
from datasets import load_dataset

import torch
import transformers
from typing import Union, Dict, Sequence
from transformers import LlamaForCausalLM,LlamaTokenizer
from peft import PeftModel

import json
import numpy as np
import argparse
import tqdm
from tqdm import tqdm

def load_model(model_name_or_path: str):
    # 加载预训练的模型和分词器(tokenizer)

    model = LlamaForCausalLM.from_pretrained(model_name_or_path, return_dict=True,output_hidden_states=True,device_map = "auto")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,padding_side="right",use_fast=False,)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.half()
    model.eval()

    # print(next(model.parameters()).device)  # 查看模型是否加载到GPU上

    return model,tokenizer

def main():

    parser = argparse.ArgumentParser(description='Generate Target with logits.')

    parser.add_argument('--model_name_or_path ', help='Model merged with lora checkpoint')
    parser.add_argument('--input_file', help='pre-data without target')
    parser.add_argument('--output_file', help='out_data with target and logits')
    parser.add_argument('--max_new_tokens', type=int, help='the length of target')

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name_or_path)

    input_file = args.input_file
    output_file = args.output_file

    folder = os.path.exists(output_file)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        f=open(output_file,'w')
        f.close()  

    prompt_column = 'input'
    response_column = 'target'

    data = []

    with open(input_file,'r') as f:
        for line in f.readlines():
            j = json.loads(line)
            data.append(j)

    for data_line in tqdm(data):
        if data_line[prompt_column]:
            query = data_line[prompt_column]
        input_ids = tokenizer(query,
                            #max_length=896,
                            truncation=True,
                            padding=True,return_tensors="pt",
                            add_special_tokens=False).input_ids



        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
                                                                
        #参数随便选择，保证有output_hidden_states，output_scores
        generate_input = {
            "input_ids":input_ids,
            "return_dict_in_generate":True,
            "output_scores":True,
            "max_new_tokens":args.max_new_tokens,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id,
        }
        #输出类型SampleDecoderOnlyOutput
        with torch.no_grad():
            output  = model.generate(**generate_input)

        #全部的输出
        #decoded_output = [tokenizer.decode(ids) for ids in output.sequences]

        # 减去输入之后，剩下的是模型的回答
        # only use id's that were generated
        # gen_sequences has shape [12, 48]
        gen_sequences = output.sequences[:, input_ids.shape[-1]:]
        res = [tokenizer.decode(ids) for ids in gen_sequences]
        res = ''.join(res).strip().rstrip('\n</s>')
        data_line['target'] = res
        #长度判断
        # tgt_len0 = output.sequences.size()[-1]-input_ids.size()[-1]#48-12
        # tgt_len = len(output.scores)#36

        # 计算并打印每个token的概率
        token_probs = []
        for i, score in enumerate(output.scores):
            # Convert the scores to probabilities
            probs = torch.softmax(score, -1)
            # Take the probability for the generated tokens (at position i in sequence)
            token_probs.append(probs[0, gen_sequences[0, i]].item())

        mean1 = np.mean(token_probs)
        data_line['probs'] = mean1
        
        with open(output_file,'a',encoding="utf-8") as cache:
            out = json.dumps(data_line, ensure_ascii=False)
            cache.write(out)
            cache.write('\n')



if __name__ == "__main__":
    main()