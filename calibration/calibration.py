import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import ast
import logging

logging.basicConfig(filename='calibration/calculate.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description='Generate Calibration')
    parser.add_argument('--task_list', type=str, required=True)
    parser.add_argument('--prob_list', type=str)
    parser.add_argument('--title', type=str)

    args = parser.parse_args()

    task_list = ast.literal_eval(args.task_list)
    prob_list = ast.literal_eval(args.prob_list)
    title = ast.literal_eval(args.title)

    plt.figure(figsize=(12,12))
    for i,task in enumerate(task_list):
        data = pd.read_csv(f'CBLUEdatasets/{task}/pre-data/embeddings_testllama-7b_0_reg_predictions.csv')
        
        # plt.rcParams.update({"font.size":20})

        logger.info("*"*22+f"START:{task}"+"*"*22+"\nTask:"+task)

        # 分组数
        num_groups = 10

        # 定义区间
        bins = np.linspace(0, 1, 11)
        x = np.linspace(0, 9, 10)
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(10)]
        y = bin_centers  # 因为x轴是区间中点，所以y的值就是bin_centers

        for idx,prob in enumerate(prob_list):

            n = len(data)
            mce = 0
            data['bins'] = np.digitize(data[prob], bins, right=True)

            # 计算每个区间的实际概率和平均预测概率
            bin_actual = 0
            bin_predicted = 0

            for i in range(1, num_groups + 1):
                bin_data = data[data['bins'] == i]
                if len(bin_data) > 0:
                    actual_prob = bin_data['label'].mean()
                    predicted_prob = bin_data[prob].mean()
                    error = abs(actual_prob-predicted_prob)*len(data)
                    bin_predicted += error
                    # bin_actual += actual_prob
                    # bin_predicted += predicted_prob
                    mce = abs(actual_prob-predicted_prob) if abs(actual_prob-predicted_prob) > mce else mce

            # 计算每个区间的ECE
            ece = bin_predicted/n

            logger.info('%-32s: %s'%(f"{prob}_MCE",str(mce)))
            logger.info('%-32s: %s'%(f"{prob}_ECE",str(ece)))

            # 将a列的值按照区间进行分组
            data['bin'] = pd.cut(data[prob], bins=bins, include_lowest=True, labels=False)

            # 计算每个区间的b列平均值,如果区间为空，设为0
            average_b = data.groupby('bin')['label'].mean()
            average_b = average_b.reindex(range(10), fill_value=0)

            #bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(10)]
            bin_labels = [f'{center:.2f}' for center in bin_centers]
            plt.subplot(3,3,i+1)
            plt.xlim((0,1))
            plt.ylim((0,1))

            #添加y=x的曲线
            plt.plot(x, y, color='red', linestyle='--', label='calibrated')

            # 绘制柱状图
            plt.bar(range(10), average_b, width=0.9, align='center', edgecolor='black')
            plt.xticks(range(10), bin_labels, rotation=45)
            plt.xlabel('prediction',fontsize=10)
            plt.ylabel('label',fontsize=10)
            plt.title(title[i])
            plt.legend(loc='upper left')

            logger.info("*"*26+"END"+"*"*26)
                
        
        plt.suptitle(f'{task} calibration',fontsize=10)
        plt.tight_layout() 
        plt.savefig(f'figs/{task}.png')
        plt.close()
        print(f'end:{task}')

if __name__ == "__main__":
    main()
