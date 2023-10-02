import os
import re

import numpy as np
import pandas as pd


def main():
    benchmark_path = './logs/logs_thesis/uniform/'
    # benchmark_path = './logs/logs_thesis/uniform_fine_tune/'
    benchmark_set = benchmark_path.split('/')[-2]
    output_path = f'./out/{benchmark_set}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folder_names = []
    for x in sorted(os.listdir(benchmark_path)):
        if os.path.isdir(os.path.join(benchmark_path, x)):
            folder_names.append(x)
    
    maps = {"names": [], "moda": [], "moda_std": [], "modp": [], "modp_std": [],
            "prec": [], "prec_std": [], "recall": [], "recall_std": []}

    for folder_name in folder_names:
        runs = sorted(os.listdir(os.path.join(benchmark_path, folder_name)))
        modas, modps, precs, recalls = [], [], [], []
        for run in runs:
            log_path = os.path.join(benchmark_path, folder_name, run, 'log.txt')
            with open(log_path, 'r') as f:
                lines = f.read()
            idx = lines.find('Test loaded model...')
            if idx == -1:
                raise ValueError('No test loaded model found')
            results = lines[idx:]

            modas.append(float(re.search(r"moda: (\d+\.\d+)%", results).group(1)))
            modps.append(float(re.search(r"modp: (\d+\.\d+)%", results).group(1)))
            precs.append(float(re.search(r"prec: (\d+\.\d+)%", results).group(1)))
            recalls.append(float(re.search(r"recall: (\d+\.\d+)%", results).group(1)))
        
        maps["names"].append(folder_name)
        maps["moda"].append(np.mean(modas))
        maps["modp"].append(np.mean(modps))
        maps["prec"].append(np.mean(precs))
        maps["recall"].append(np.mean(recalls))
        maps["moda_std"].append(np.std(modas, ddof=1))
        maps["modp_std"].append(np.std(modps, ddof=1))
        maps["prec_std"].append(np.std(precs, ddof=1))
        maps["recall_std"].append(np.std(recalls, ddof=1))
    
    df = pd.DataFrame(maps)
    df.to_csv(os.path.join(output_path, 'analysis_results.csv'), index=False)


if __name__ == "__main__":
    main()
