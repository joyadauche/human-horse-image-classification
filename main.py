import tensorflow as tf
import pathlib
import os
from shutil import copyfile
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_SIZE = 100

_, info = tfds.load('horses_or_humans', with_info=True)
print(info)
print(info.features['image'].shape)

train_data = tfds.load('horses_or_humans', split='train')
valid_data = tfds.load('horses_or_humans', split='test')

train_data_len = len(list(train_data))
val_data_len = len(list(valid_data))
print(train_data_len)
print(val_data_len)


def format_image(features):
    image = features['image']
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, features['label']


train_batches = train_data.shuffle(NUM_SIZE).map(format_image).batch(BATCH_SIZE)
validation_batches = valid_data.map(format_image).batch(BATCH_SIZE)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.95:
            print('\n Reached 99% accuracy, so cancelling training')
            self.model.stop_training = True


callbacks = MyCallback()


pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights('./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=10,
    callbacks=[callbacks]
)
model.save("human_horse_classifier.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
print(acc)
print(val_acc)

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()
