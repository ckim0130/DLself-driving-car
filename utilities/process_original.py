import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import cv2
from natsort import natsorted
import glob
import pandas as pd

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import cv2
from natsort import natsorted
import glob
import pandas as pd

def add_label(row):
    label = np.array([row["angle"], row["speed"]])
    return label

if __name__ == "__main__":
    
    train_images_path = 'machine-learning-in-science-2021/training_data/training_data/*.png'
    eval_images_path = 'machine-learning-in-science-2021/test_data/test_data/*.png'
    train_image_paths = natsorted(glob.glob(train_images_path))
    eval_image_paths = natsorted(glob.glob(eval_images_path))
    csv_path = 'machine-learning-in-science-2021/training_norm.csv'
    eval_path = 'machine-learning-in-science-2021/eval_data/*.png'

    ptrain_images_path = 'processed_dataset/training_data/train_images.npy'
    peval_images_path = 'processed_dataset/eval_data/eval_images.npy'
    plabels_path = 'processed_dataset/training_data/labels.npy'
    
    train_images = []
    corrupted = []

    for index, file in enumerate(train_image_paths):
        image = cv2.imread(file)
        if image is not None:
            train_images.append(cv2.imread(file)) 
        else:
            corrupted.append(index)
    print(corrupted)
            
    train_images = np.asarray(train_images)
    np.save(ptrain_images_path, train_images)
    
    eval_images = []
    for file in eval_image_paths:
        eval_images.append(cv2.imread(file))    
    eval_images = np.asarray(eval_images)
    np.save(peval_images_path, eval_images)
    
    
    df = pd.read_csv(csv_path)
    df = df.drop(df.index[corrupted])
    df['label'] = df.apply(lambda row: add_label(row), axis=1)
    labels = np.array(df['label'].values)
    np.save(plabels_path, labels)