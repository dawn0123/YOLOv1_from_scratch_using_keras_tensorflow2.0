from dataset import ImageGenerator
import os
from config import DATA_PATH, batch_size
from model import get_model
from lrscheduler import LearningRateScheduler, lr_schedule
from loss import yolo_loss
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf


num_train_samples = 0
num_valid_samples = 0


def makeImageTrainValidGenerator():
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

    my_training_batch_generator = ImageGenerator(X_train, Y_train, batch_size)
    my_validation_batch_generator = ImageGenerator(X_val, Y_val, batch_size)
    global num_train_samples, num_valid_samples
    num_train_samples = len(X_train)
    num_valid_samples = len(X_val)

    return my_training_batch_generator, my_validation_batch_generator


def train():
    train_gen, valid_gen = makeImageTrainValidGenerator()
    model = get_model()
    mcp_save = ModelCheckpoint(
        'weight.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.compile(loss=yolo_loss, optimizer='adam')
    model.fit(
        x=train_gen,
        steps_per_epoch=int(num_train_samples // batch_size),
        epochs=135,
        verbose=1,
        workers=4,
        validation_data=valid_gen,
        validation_steps=int(num_valid_samples // batch_size),
        callbacks=[
            LearningRateScheduler(lr_schedule),
            mcp_save
        ]
    )


if __name__ == '__main__':
    train()
