import tensorflow as tf
import pandas as pd
import numpy as np


class DS_Reader():
    def __init__(self):
        self.total_examples = 0
        self.dataset = 0
        
    def read(self, outer_path = None):
        
        if outer_path == None:
            outer_path = ''

        training_path = outer_path + 'processed_dataset/training_data/'
        eval_path = outer_path + 'processed_dataset/eval_data/'

        labels_path = training_path + 'labels.npy'
        train_images_path = training_path + 'train_images.npy'
        eval_images_path = eval_path + 'eval_images.npy'


        train_images = np.load(train_images_path, allow_pickle = True)
        labels = np.load(labels_path, allow_pickle = True)
        eval_images = np.load(train_images_path, allow_pickle = True)
        eval_images = np.load(eval_images_path, allow_pickle = True)
        

        self.total_examples = labels.shape[0]


        self.train_chunks = list(train_images)
        self.label_chunks = list(labels)

        self.train_dataset = tf.data.Dataset.from_generator(self.train_genenerator, output_signature=(
                                     tf.TensorSpec(shape=(240, 320, 3), dtype=tf.float32),
                                     tf.TensorSpec(shape=(2,), dtype=tf.float32)))
        self.eval_dataset = tf.cast(tf.constant(eval_images), dtype=tf.float32)


    def get_datasets(self, batch_size, shuffle_size, train_fraction):
        train_examples = int(self.total_examples*train_fraction)
        dataset_train = self.train_dataset.take(train_examples).shuffle(shuffle_size).batch(batch_size, drop_remainder=True)
        dataset_test = self.train_dataset.skip(train_examples).shuffle(shuffle_size).batch(batch_size, drop_remainder=True)
        dataset_eval = self.eval_dataset
        return dataset_train, dataset_test, dataset_eval
    
    def train_genenerator(self):
        for i, j in zip(self.train_chunks, self.label_chunks):
            yield i, j
