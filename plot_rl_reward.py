import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    path = './logs/logs_thesis/rl_training_reward_plot/'
    output_path = './out/rl_training_reward_plot/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = []

    for run in sorted(os.listdir(path)):
        results.append(np.load(os.path.join(path, run, 'step_rewards.npy')))
    
    # get results, sample for every 10 steps
    results = np.stack(results).squeeze()
    print(results.shape)
    mask = np.arange(0, results.shape[1], 10)
    results = results[:, mask]
    print(results.shape)

    df = pd.DataFrame(results).melt(var_name="Timesteps", value_name="Reward")

    plt.figure(figsize=(10,6))
    sns.lineplot(x="Timesteps", y="Reward", data=df, errorbar='sd')
    plt.title("Plot of reward versus timesteps across runs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'rl_training_reward_plot.pdf'))


if __name__ == '__main__':
    main()
