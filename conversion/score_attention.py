import pandas as pd
import ast 
import argparse


def main():

    parser = argparse.ArgumentParser(description='Generate from Attention')
    parser.add_argument('--layers', type=str, required=True,help='a list of integers representing layers')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task_list', type=str)

    args = parser.parse_args()

    # 将参数字符串解析为列表
    try:
        layers = ast.literal_eval(args.layers)
        if not isinstance(layers, list):
            raise argparse.ArgumentTypeError("Layers must be a list")
        if not all(isinstance(layer, int) for layer in layers):
            raise argparse.ArgumentTypeError("All elements in layers must be integers")
    except ValueError:
        parser.error("Invalid layers format. Please provide a valid list format.")
    task_list = ast.literal_eval(args.task_list)
    data_dir = args.data_dir

    for layer in layers:
        for task in task_list:
            # 加载CSV文件
            file_path = f"{data_dir}/embeddings_{task}llama-7b_{layer}.csv"  # 替换为实际文件路径
            df = pd.read_csv(file_path)

            # 将attentions列从字符串表示转换为实际的浮点数列表
            df['attentions'] = df['attentions'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

            # 找到最短的attentions列表的长度
            min_length = df['attentions'].apply(len).min()
            print(f"最短的attentions列表的长度: {min_length}")

            # 创建新的列max_a, min_a, 和 abs_a
            df['max_a'] = df['attentions'].apply(lambda x: sorted(x, reverse=True)[1:11])
            df['min_a'] = df['attentions'].apply(lambda x: sorted(x)[1:11])
            df['abs_a'] = df.apply(lambda row: [abs(m - n) for m, n in zip(row['max_a'], row['min_a'])], axis=1)

            # 保存修改后的DataFrame到新的CSV文件
            new_file_path = f"{data_dir}/attentions_testllama-7b_{layer}.csv"  # 替换为实际保存路径
            df.to_csv(new_file_path, index=False)

            print(f"处理后的文件已保存为: {new_file_path}")
