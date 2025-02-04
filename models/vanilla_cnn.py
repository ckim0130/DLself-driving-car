from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.python.keras import regularizers

class V_CNN(Model):
  def __init__(self):
    super(V_CNN, self).__init__()
    l = 0.0001
    self.batch0 = BatchNormalization()
    self.conv1 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l))
    self.conv2 = Conv2D(16, (3, 3), activation='relu')
    self.conv3 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l))
    self.drop1 = Dropout(0.2)
    self.flatten = Flatten()
    self.batch1 = BatchNormalization()
    self.d1 = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l))
    self.d2 = Dense(2, activation='relu')
    self.batch2 = BatchNormalization()
    self.d3 = Dense(2, activation='relu', kernel_regularizer=regularizers.l2(l))

  def call(self, x):
    x = self.b
    x = self.conv1(x)
    x = self.drop1(x)
    x = self.flatten(x)
    x = self.batch1(x)
    x = self.d1(x)
    x = self.batch2(x)
    x = self.d2(x)
    return x