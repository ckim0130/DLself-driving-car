from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.python.keras import regularizers

class NV_CNN(Model):
  def __init__(self):
    super(NV_CNN, self).__init__()
    l = 0.0001
    self.batch0 = BatchNormalization()
    self.conv1 = Conv2D(24, (5, 5), activation='relu', strides = (2, 2), kernel_regularizer=regularizers.l2(l))
    self.conv2 = Conv2D(36, (5, 5), activation='relu', strides = (2, 2))
    self.conv3 = Conv2D(48, (5, 5), activation='relu', strides = (2, 2), kernel_regularizer=regularizers.l2(l))
    self.conv4 = Conv2D(64, (3, 3), activation='relu', strides = (1, 1))
    self.conv5 = Conv2D(64, (3, 3), activation='relu', strides = (1, 1), kernel_regularizer=regularizers.l2(l))
    self.flatten = Flatten()
    self.batch1 = BatchNormalization()
    self.d1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l))
    self.d2 = Dense(50, activation='relu')
    self.batch2 = BatchNormalization()
    self.d3 = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l))
    self.d4 = Dense(2, activation='relu', kernel_regularizer=regularizers.l2(l))

  def call(self, x):
    x = self.batch0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.flatten(x)
    x = self.batch1(x)
    x = self.d1(x)
    x = self.batch2(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    return x