# https://keraskorea.github.io/posts/2018-10-23-keras_autoencoder/

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np
#from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

# size of encoded representation
encoding_dim = 600

# Input place holder
input_img = Input(shape=(60 * 36 * 3,))

# input -> encoded
encoded = Dense(encoding_dim, activation='relu')(input_img)

# encoded -> decoded
decoded = Dense(60 * 36* 3, activation='sigmoid')(encoded)

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


x_train1 = []
x_test1  = []

for i in range(1, 14):
    dataset_path = "data_set/"
    name = "normal_" + str(i).zfill(4) + "_re.jpg"
    print(name)
    #img = image.load_img(dataset_path + name)
    img = cv2.imread(dataset_path + name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (60, 36))
    result_path = "result_set/"
    cv2.imwrite(result_path + name, img)

    img = img.reshape(1, 60 * 36 * 3)
    x_train1.append(img)
    x_test1.append(img)

    #x_train = np.concatenate(x_train, img)
    #x_test = np.concatenate(x_train, img)
    

x_train = np.array(x_train1)
x_test = np.array(x_test1)
#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=1000,
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
    RGB_img = cv2.cvtColor(x_test[i].reshape(36, 60, 3), cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # recounstructed output
    ax = plt.subplot(2, n, i + 1 + n)
    RGB_img = cv2.cvtColor(decoded_imgs[i].reshape(36, 60, 3), cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
