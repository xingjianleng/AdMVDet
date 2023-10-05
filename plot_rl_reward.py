import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    # set seaborn fonts larger
    sns.set_context("notebook", font_scale=1.25)

    path = './logs/logs_thesis/rl_training_reward_plot/delta_moda/'
    # path = './logs/logs_thesis/rl_training_reward_plot/delta_loss/'
    method = path.split('/')[-2]
    output_path = './out/rl_training_reward_plot/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = []

    for run in sorted(os.listdir(path)):
        results.append(np.load(os.path.join(path, run, 'step_rewards.npy')))
    
    # get results, sample for every 10 steps
    results = np.stack(results).squeeze()

    df = pd.DataFrame(results).melt(var_name="Timesteps", value_name="Reward")

    plt.figure(figsize=(10,7))
    sns.lineplot(x="Timesteps", y="Reward", data=df, errorbar='sd', color='green', label='Stepwise rewards')
    sns.regplot(x="Timesteps", y="Reward", data=df, scatter=False, color='red', label='Regression line')
    # plt.title(f"Plot of reward versus timesteps across runs with {method} method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'rl_training_reward_plot_{method}.pdf'))


if __name__ == '__main__':
    main()
