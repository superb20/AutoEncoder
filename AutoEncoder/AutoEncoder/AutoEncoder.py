# https://keraskorea.github.io/posts/2018-10-23-keras_autoencoder/

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

# size of encoded representation
encoding_dim = 32

# Input place holder
input_img = Input(shape=(784,))

# input -> encoded
encoded = Dense(encoding_dim, activation='relu')(input_img)

# encoded -> decoded
decoded = Dense(784, activation='sigmoid')(encoded)

# input -> encoded -> decoded
autoencoder = Model(input_img, decoded)

# input -> encoded
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# last layer of autoencoder
decoder_layer = autoencoder.layers[-1]

# encoded -> decoded
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# use test data
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # origin data
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # recounstructed output
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
