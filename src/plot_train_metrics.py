import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
plt.rcParams.update(rcParams)
sns.set_theme(style="darkgrid")


def plot_training_metrics_normorder(model, model_name, dataset_name, runs=5, epochs=50):
    experiments = ['nl_nr', 'l1_ur', 'l2_ur', 'li_ur', 'll_ur']
    experiments_label = ['No Norm', 'L1, unit r', 'L2, unit r', 'L-inf, unit r', 'L-learn, unit r']
    metrics = ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', 'loss', 'val_loss']
    metrics_label = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'black']
    markers = ["o", "^", ">", "D", "<"]
    for idx_metric, metric in enumerate(metrics):
        fig, ax = plt.subplots()
        for idx_experiment, experiment in enumerate(experiments):
            exp_runs_metric = np.empty((0, epochs))
            for run in range(runs):
                exp_run = json.load(open(f'results/{model}/{dataset_name}/th_{model}_{experiment}_run{run}.json'))
                exp_runs_metric = np.vstack((exp_runs_metric, exp_run[metric]))
            x = np.array([*range(50)])
            y = np.mean(exp_runs_metric, axis=0)
            ci = stats.t.interval(0.9, len(exp_runs_metric) - 1, loc=np.mean(exp_runs_metric, axis=0),
                                  scale=stats.sem(exp_runs_metric, axis=0))
            ax.plot(x, y, label=experiments_label[idx_experiment], color=colors[idx_experiment],
                    marker=markers[idx_experiment], markevery=10)
            ax.fill_between(x, ci[0], ci[1], color=colors[idx_experiment], alpha=.1)
        ax.legend()
        plt.title(f'{metrics_label[idx_metric]} - {model_name} - {dataset_name}')
        plt.xlabel('epochs')
        plt.ylabel(f'{metrics[idx_metric]}')
        fig.savefig(f'./figures/{model}/{dataset_name}/normorder_{metric}.png')
        plt.close()


def plot_training_metrics_normradius(model, model_name, dataset_name, runs=5, epochs=50):
    experiments = ['l2_ur', 'l2_lr', 'll_ur', 'll_lr']
    experiments_label = ['L2, unit r', 'L2, learn r', 'L-learn, unit r', 'L-learn, learn r']
    metrics = ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', 'loss', 'val_loss']
    metrics_label = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'black']
    markers = ["o", "^", ">", "D", "<"]
    for idx_metric, metric in enumerate(metrics):
        fig, ax = plt.subplots()
        for idx_experiment, experiment in enumerate(experiments):
            exp_runs_metric = np.empty((0, epochs))
            for run in range(runs):
                exp_run = json.load(
                    open(f'results/{model}/{dataset_name}/th_{model}_{experiment}_run{run}.json'))
                exp_runs_metric = np.vstack((exp_runs_metric, exp_run[metric]))
            x = np.array([*range(50)])
            y = np.mean(exp_runs_metric, axis=0)
            ci = stats.t.interval(0.9, len(exp_runs_metric) - 1, loc=np.mean(exp_runs_metric, axis=0),
                                  scale=stats.sem(exp_runs_metric, axis=0))
            ax.plot(x, y, label=experiments_label[idx_experiment], color=colors[idx_experiment],
                    marker=markers[idx_experiment], markevery=10)
            ax.fill_between(x, ci[0], ci[1], color=colors[idx_experiment], alpha=.1)
        ax.legend()
        plt.title(f'{metrics_label[idx_metric]} - {model_name} - {dataset_name}')
        plt.xlabel('epochs')
        plt.ylabel(f'{metrics[idx_metric]}')
        fig.savefig(f'./figures/{model}/{dataset_name}/normradius_{metric}.png')
        plt.close()


def plot_train_metrics(model, model_name, dataset_name, epochs=50, runs=5):
    # path to store plots of training metrics
    save_directory = f"./figures/{model}/{dataset_name}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plot_training_metrics_normorder(model, model_name, dataset_name, epochs=epochs, runs=runs)
    plot_training_metrics_normorder(model, model_name, dataset_name, epochs=epochs, runs=runs)
