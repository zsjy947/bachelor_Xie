"""
Embedding Generation for Language Model for LLAMA and OPT models
Date: 2023-05-26


This script loads sentences from specified CSV files, processes them with a specified LLaMa or OPT model with Hugging Face's transformers library,
and saves the embeddings of the last token of each sentence into new CSV files.

It does the same work as Generate_Embeddings.py but adds (fragile) functionality for LLaMA.

It's based on Amos Azaria's and Tom Mitchell's implementation for their paper `The Internal State of an LLM Knows When it's Lying.' 
https://arxiv.org/abs/2304.13734

It uses the Hugging Face's tokenizer, with the model names specified in a configuration JSON file or by commandline args. 
Model options for OPT include: '6.7b', '2.7b', '1.3b', '350m', 
Model options for LLaMa include: '7B', '13B', '30B', and '65B'. 

The configuration file and/or commandline args also specify whether to remove periods at the end of sentences, which layers of the model to use for generating embeddings,
and the list of datasets to process.

!!!!!!
CAUTION: Because the LLaMa models are not fully publically available, paths for loading those models are hard-coded into the `load_llama_model` function.
!!!!!

If any step fails, the script logs an error message and continues with the next dataset.

Requirements:
- transformers library
- pandas library
- numpy library
- pathlib library
"""
import sys
sys.path.append('./')
import torch
from transformers import AutoTokenizer, OPTForCausalLM, LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 这一行注释掉就是使用cpu，不注释就是使用gpu。

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='embedding_extraction.log')

def load_llama_model(model_path: str):
    '''
    Initializes and returns a LLaMa model and tokenizer.

    Args:
    model_name: str. A string representing the model name. Must be one of '7B', '13B', or '30B'.

    Returns:
    Tuple[LlamaForCausalLM, LlamaTokenizer]. A tuple containing the loaded LLaMa model and its tokenizer.
    '''

    tokenizer = LlamaTokenizer.from_pretrained(model_path,max_legnth=512, return_attention_mask=True)
    model = LlamaForCausalLM.from_pretrained(model_path,device_map = "auto")
    model = model.half()
    print(next(model.parameters()).device)
    return model, tokenizer


def init_model(model_name: str,model_path: str):
    """c
    Initializes and returns the model and tokenizer.
    """
    try:
        if model_name in ['llama-7b', '13B', '30B']:
            model, tokenizer = load_llama_model(model_path)
        else:
            model = OPTForCausalLM.from_pretrained("facebook/opt-"+model_name)
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+model_name)
    except Exception as e:
        print(f"An error occurred when initializing the model: {str(e)}")
        return None, None
    return model, tokenizer

#def load_data(dataset_path: Path, dataset_name: str, true_false: bool = False):
def load_data(dataset_path: Path, dataset_name: str, true_false:bool):
    filename_suffix = "_true_false" if true_false else ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    try:
        df = pd.read_csv(dataset_file,encoding='utf-8')
    except FileNotFoundError as e:
        print(f"Dataset file {dataset_file} not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file {dataset_file}: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"No data in CSV file {dataset_file}: {str(e)}")
        return None
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype='object')
    return df

def compute_ati(attention_matrix):
    L = attention_matrix.shape[0]
    ati_values = []

    for i in range(L):
        left_and_self_sum = sum(attention_matrix[i, :i+1])  # 包含 attention_matrix[i, i]
        right_sum = sum(attention_matrix[j, i] for j in range(i+1, L))
        ati = left_and_self_sum + right_sum
        ati_values.append(ati)

    return ati_values

def process_row(prompt: str, model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a row of data and returns the embeddings.
    """
    if remove_period:
        prompt = prompt.rstrip(". ")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
    embeddings = {}
    for layer in layers_to_use:
        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        embeddings[layer] = [last_hidden_state.numpy().tolist()]
    return embeddings

#Still not convinced this function works 100% correctly, but it's much faster than process_row.
def process_batch(batch_prompts: List[str], model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a batch of data and returns the embeddings for each statement.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or any other token of your choice
    
    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    batch_size = inputs['input_ids'].shape[0]

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True,output_attentions=True) 
    
    # Use the attention mask to find the index of the last real token for each sequence
    seq_lengths = inputs.attention_mask.sum(dim=1) - 1  # Subtract 1 to get the index

    batch_embeddings = {}
    for layer in layers_to_use:
        hidden_states = outputs.hidden_states[layer]

        # Gather the hidden state at the last real token for each sequence
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.detach().cpu().numpy().tolist() for embedding in last_hidden_states]
    
    batch_ati = []
    for layer in layers_to_use:
        layer_attention = outputs.attentions[layer]
        layer_ati = []
        for batch_idx in range(batch_size):
            all_attention = np.zeros(layer_attention[0,0].shape)
            for head_idx in range(layer_attention.size(1)):
                head_attention = layer_attention[batch_idx,head_idx].squeeze().detach().cpu().numpy()
                all_attention += head_attention
                seq_length = (inputs['attention_mask'][batch_idx] == 1).sum().item()
            all_attention = all_attention / layer_attention.size(1)
            all_attention = all_attention[:seq_length,:seq_length]
            ave_ati = compute_ati(all_attention)
            layer_ati.append(ave_ati)
        batch_ati.append(layer_ati)

    return batch_embeddings,batch_ati


def save_data(df, output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    """
    Saves the processed data to a CSV file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"attentionss_{dataset_name}{model_name}_{abs(layer)}{filename_suffix}.csv"
    try:
        df.to_csv(output_file, index=False,encoding='utf-8')
    except PermissionError:
        print(f"Permission denied when trying to write to {output_file}. Please check your file permissions.")
    except Exception as e:
        print(f"An unexpected error occurred when trying to write to {output_file}: {e}")


def main():
    """
    Loads configuration parameters, initializes the model and tokenizer, and processes datasets.

    Configuration parameters are loaded from a JSON file named "BenConfigMultiLayer.json". 
    These parameters specify the model to use, whether to remove periods from the end of sentences, 
    which layers of the model to use for generating embeddings, the list of datasets to process, 
    and the paths to the input datasets and output location.

    The script processes each dataset according to the configuration parameters, generates embeddings for 
    each sentence in the dataset using the specified model and layers, and saves the processed data to a CSV file. 
    If processing a dataset or saving the data fails, the script logs an error message and continues with the next dataset.
    """
    try:
        with open("/calibration/config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
        return
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
        return
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")
        return

    '''
    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--model", 
                        help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--layers", nargs='*', 
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*',
                        help="List of dataset names without csv extension. Can leave off 'true_false' suffix if true_false flag is set to True")
    parser.add_argument("--true_false", action="store_true", help="Do you want to append 'true_false' to the dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", action="store_true", help="Include this flag if you want to extract embedding for the last token before the final period.")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    true_false = args.true_false if args.true_false is not None else config_parameters["true_false"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else config_parameters["batch_size"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])
    '''
    model_name = config_parameters["model"]
    should_remove_period = config_parameters["remove_period"]
    layers_to_process = config_parameters["layers_to_use"]
    dataset_names = config_parameters["list_of_datasets"]
    true_false =  config_parameters["true_false"]
    BATCH_SIZE = config_parameters["batch_size"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])
    model_path = config_parameters["model_path"]

    model_output_per_layer: Dict[int, pd.DataFrame] = {}

    model, tokenizer = init_model(model_name,model_path)
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return
    #I've left this in in case there's an issue with the batch_processing fanciness
    # for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
    #     dataset = load_data(dataset_path, dataset_name, true_false=true_false)
    #     if dataset is None:
    #         continue
    #     for layer in layers_to_process:
    #         model_output_per_layer[layer] = dataset.copy()

    #     for i, row in tqdm(dataset.iterrows(), desc="Row number"):
    #         sentence = row['statement']
    #         embeddings = process_row(sentence, model, tokenizer, layers_to_process, should_remove_period)
    #         for layer in layers_to_process:
    #             model_output_per_layer[layer].at[i, 'embeddings'] = embeddings[layer]
    #         if i % 100 == 0:
    #             logging.info(f"Processing row {i}")

    #     for layer in layers_to_process:
    #         save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period) 

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        # Increase the threshold parameter to a large number
        np.set_printoptions(threshold=np.inf)
        dataset = load_data(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_batches = len(dataset) // BATCH_SIZE + (len(dataset) % BATCH_SIZE != 0)

        for layer in layers_to_process:
            model_output_per_layer[layer] = dataset.copy()
            model_output_per_layer[layer]['embeddings'] = pd.Series(dtype='object')
            model_output_per_layer[layer]['attentions'] = pd.Series(dtype='object')

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            start_idx = batch_num * BATCH_SIZE
            actual_batch_size = min(BATCH_SIZE, len(dataset) - start_idx)
            end_idx = start_idx + actual_batch_size
            batch = dataset.iloc[start_idx:end_idx]
            batch_prompts = batch['statement'].tolist()

            batch_embeddings,batch_ati = process_batch(batch_prompts, model, tokenizer, layers_to_process, should_remove_period)

            for layer in layers_to_process:
                for i, idx in enumerate(range(start_idx, end_idx)):
                    model_output_per_layer[layer].at[idx, 'embeddings'] = batch_embeddings[layer][i]

            for layer,layer_ati in zip(layers_to_process,batch_ati):
                for i,idx in  enumerate(range(start_idx, end_idx)):
                    model_output_per_layer[layer].at[idx,'attentions'] = layer_ati[i]

            if batch_num % 10 == 0:
                logging.info(f"Processing batch {batch_num}")

        for layer in layers_to_process:
            save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period)


if __name__ == "__main__":
    main()
