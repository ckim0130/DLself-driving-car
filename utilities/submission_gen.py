import pandas as pd
import tensorflow as tf
import numpy as np


def speed_to_bin(row):
    speed = row['speed'] 
    if abs(speed - 1) < abs(speed):
        speed = 1
    else:
        speed = 0      
    return speed


class SubmissionGen():
    def __init__(self):
        self.predictions = None
        
    def make_csv(self, predictions):
        columns = ['image_id', 'angle', 'speed']
        index = np.arange(1, predictions.shape[0] + 1)

        angle = tf.transpose(predictions)[0][:].numpy()  
        angle_norm = angle/np.amax(angle)

        speed = tf.transpose(predictions)[1][:].numpy()
        speed_norm = speed/np.amax(speed)
        


        df = pd.DataFrame(columns = columns)

        df['image_id'] = index
        df['angle'] = angle_norm
        df['speed'] = speed_norm
        df['speed'] = df.apply(lambda x: speed_to_bin(x), axis=1)

        return df