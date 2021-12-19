# image classification datasets from tensorflow_datasets catalog
list_of_datasets = [
    {"name": "beans", "train_split": "train", "validation_split": "validation", "num_classes": 3, "IMG_SHAPE": 224,
     "batch_size": 64, "learning_rate": 0.0001},
    {"name": "cats_vs_dogs", "train_split": "train[:85%]", "validation_split": "train[85%:]", "num_classes": 2,
     "IMG_SHAPE": 128, "batch_size": 64, "learning_rate": 0.0001},
    {"name": "cifar10", "train_split": "train", "validation_split": "test", "num_classes": 10, "IMG_SHAPE": 32,
     "batch_size": 128, "learning_rate": 0.0001},
    {"name": "cifar100", "train_split": "train", "validation_split": "test", "num_classes": 100, "IMG_SHAPE": 32,
     "batch_size": 128, "learning_rate": 0.0001},
    {"name": "imagenette", "train_split": "train", "validation_split": "validation", "num_classes": 10,
     "IMG_SHAPE": 128, "batch_size": 64, "learning_rate": 0.0001},
    {"name": "rock_paper_scissors", "train_split": "train", "validation_split": "test", "num_classes": 3,
     "IMG_SHAPE": 224, "batch_size": 32, "learning_rate": 0.0001},
    {"name": "stanford_dogs", "train_split": "train", "validation_split": "test", "num_classes": 120, "IMG_SHAPE": 128,
     "batch_size": 64, "learning_rate": 0.0001},
    {"name": "tf_flowers", "train_split": "train[:85%]", "validation_split": "train[85%:]", "num_classes": 5,
     "IMG_SHAPE": 224, "batch_size": 32, "learning_rate": 0.0001}
]

# order, radius
list_of_experiments = [['nl', 'nr'],
                       ['l1', 'ur'],
                       ['l2', 'ur'],
                       ['li', 'ur'],
                       ['ll', 'ur'],
                       ['l2', 'lr'],
                       ['ll', 'lr']]

# resnet50, efficientnet-b0, visual transformer
list_of_models = ['rn5', 'ef0', 'vit']
list_of_model_names = ['ResNet50', 'EfficientNet-B0', 'Visual Transformer']


def select_dataset(dataset_name='beans'):
    for dataset in list_of_datasets:
        if dataset['name'] == dataset_name:
            return dataset


def list_available_datasets():
    dataset_names = []
    for dataset in list_of_datasets:
        dataset_names.append(dataset['name'])
    return dataset_names
