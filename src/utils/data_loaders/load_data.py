from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

def limit_data(config, n=100000):
    a = []
    
    rgb_dir = config.dataset_path

    print(f"RGB Directory: {rgb_dir}")

    files_list = os.listdir(rgb_dir)
    random.shuffle(files_list)

    for folder_name in files_list:
        rgb_folder = os.path.join(rgb_dir, folder_name)

        if not os.path.isdir(rgb_folder):
            print(f"Missing folder for class '{folder_name}'. Check RGB directories.")
            continue

        rgb_files = os.listdir(rgb_folder)
        for k, rgb_file in enumerate(rgb_files):
            if k >= n:
                break

            rgb_path = os.path.join(rgb_folder, rgb_file)

            a.append((rgb_path, folder_name))

    df = pd.DataFrame(a, columns=['rgb', 'class'])
    print(f"Total image found: {len(df)}")
    return df


class MultiModalDataGenerator(Sequence):
    def __init__(self, df, config, subset):
        self.df = df
        self.batch_size = config.batch_size
        self.target_size = (config.img_height, config.img_width)
        self.subset = subset
        self.config = config
    

        if subset == 'training':
            self.df = self.df.sample(frac=1-config.val_split, random_state=config.seed)

        elif subset == 'validation':
            self.df = self.df.drop(self.df.sample(frac=1-config.val_split, random_state=config.seed).index)
        self.log_image_counts()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))
    def log_image_counts(self):
        self.num_images = len(self.df)
        counts_per_set = self.df['class'].value_counts()
        print(f"Total image sets in {self.subset} set:")
        print(counts_per_set)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        rgb_images = []

        for _, row in batch_df.iterrows():
            rgb_img = tf.io.read_file(row['rgb'])
            rgb_img = tf.image.decode_png(rgb_img, channels=3)
            rgb_img = tf.image.resize(rgb_img, self.target_size)
            rgb_images.append(rgb_img)


        inputs = []

        rgb_images = tf.stack(rgb_images) / 255.0
        inputs.append(rgb_images)


        class_labels = {'Arborio': 0, 'Basmati': 1, 'Ipsala': 2, 'Jasmine': 3, 'Karacadag': 4}
        
        # Convert class labels to integers
        int_labels = batch_df['class'].replace(class_labels).values

        # Convert integer labels to one-hot encoding
        targets = to_categorical(int_labels, num_classes=self.config.num_classes)

        return inputs, targets

    def reset(self):
        self.df = self.df.sample(frac=1) if self.subset == 'training' else self.df
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.subset == 'training':
            np.random.shuffle(self.indexes)


def load_data(config):
    full_df = limit_data(config, config.num_img_lim)

    train_generator = MultiModalDataGenerator(full_df, config, subset='training')
    validation_generator = MultiModalDataGenerator(full_df, config, subset='validation')

    return train_generator, validation_generator
