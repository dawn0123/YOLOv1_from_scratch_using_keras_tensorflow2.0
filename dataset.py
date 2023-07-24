from tensorflow import keras
import numpy as np
from image import read


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx+1) * self.batch_size]

        train_image = []
        train_label = []

        for i in range(0, len(batch_x)):
            img_path = batch_x[i]
            label = batch_y[i]
            image, label_matrix = read(img_path, label)
            train_image.append(image)
            train_label.append(label_matrix)
        return np.array(train_image), np.array(train_label)
