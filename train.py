from src.train_resnet import train_resnet
from src.train_efficientnet import train_efficientnet
from src.train_vit import train_vit
from src.plot_train_metrics import plot_train_metrics
from src.utils import *
import os


def main():
    # select dataset
    dataset_name = list_of_datasets[5]['name']
    assert dataset_name in list_available_datasets()  # use with argparse
    dataset = select_dataset(dataset_name=dataset_name)

    # select model
    model = list_of_models[0]
    assert model in list_of_models  # use with argparse

    # path to store training results
    save_directory = f"./results/{model}/{dataset_name}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # select experiment
    order, radius = list_of_experiments[0]
    assert [order, radius] in list_of_experiments  # use with argparse
    print(f"Running experiment for model = {model} - dataset - {dataset_name} - order = {order} - radius = {radius}")
    print(f'Results will be saved at this path: {save_directory}')

    if model == 'rn5':
        train_resnet(dataset, run_exp_count=5, order='l2', radius='lr', logdir=save_directory)
        train_resnet(dataset, run_exp_count=5, order='ll', radius='lr', logdir=save_directory)
    elif model == 'ef0':
        train_efficientnet(dataset, run_exp_count=5, order=order, radius=radius, logdir=save_directory)
    elif model == 'vit':
        train_vit(dataset, run_exp_count=5, order=order, radius=radius, logdir=save_directory)


def plot():
    # select dataset
    dataset_name = list_of_datasets[5]['name']
    assert dataset_name in list_available_datasets()  # use with argparse

    # select model
    model = list_of_models[0]
    model_name = list_of_model_names[0]
    assert model in list_of_models  # use with argparse

    plot_train_metrics(model, model_name, dataset_name)


if __name__ == "__main__":
    main()
    # plot()
