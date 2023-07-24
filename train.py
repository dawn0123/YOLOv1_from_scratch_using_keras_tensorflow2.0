from dataset import ImageGenerator
import os
from config import DATA_PATH


def train():
    train_datasets = []
    val_datasets = []

    with open(os.path.join(DATA_PATH, "VOCdevkit", 'train.txt'), 'r') as f:
        train_datasets = train_datasets + f.readlines()

    with open(os.path.join(DATA_PATH, "VOCdevkit", 'val.txt'), 'r') as f:
        val_datasets = val_datasets + f.readlines()

    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    for item in train_datasets:
        item = item.replace("\n", "").split(" ")
        X_train.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_train.append(arr)

    for item in val_datasets:
        item = item.replace("\n", "").split(" ")
        X_val.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_val.append(arr)

    batch_size = 4
    my_training_batch_generator = ImageGenerator(X_train, Y_train, batch_size)
    my_validation_batch_generator = ImageGenerator(X_val, Y_val, batch_size)

    x_train, y_train = my_training_batch_generator.__getitem__(0)
    x_val, y_val = my_validation_batch_generator.__getitem__(0)
    print(x_train.shape)
    print(y_train.shape)

    print(x_val.shape)
    print(y_val.shape)


if __name__ == '__main__':
    train()
