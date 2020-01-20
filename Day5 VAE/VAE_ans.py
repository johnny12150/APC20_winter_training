from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Lambda, Reshape, ReLU
import keras.backend as K
from keras.losses import mse, binary_crossentropy

(x, _), _ = mnist.load_data()
img_rows = 64
img_cols = 64
channels = 1
img_size = (img_rows, img_cols, channels)
latent_dim = 98

x_train = x[:20000].astype('float32')/ 255.
x_test = x[20000:25000].astype('float32')/ 255.
x_train = x_train.reshape((-1, 28, 28, 1))  # reshpae for input (only 1 color)
x_test = x_test.reshape((-1, 28, 28, 1))


def compare(data, n=10):
    plt.figure(figsize=(n*2, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.close()


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


input_img = Input(shape=(28,28,1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = ReLU()(x)
h = Flatten()(x)
# add Dense of mu & sigma
z_mu = Dense(latent_dim, name='z_mu')(h)
z_log_var = Dense(latent_dim, name='z_logvar')(h)
# custom layer
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mu, z_log_var])
# reshape Input to decoder, 98/(7*7) = 2
encoded = Reshape((7, 7, 2))(z)

# decoder, same as AE
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (1, 1), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
vae = Model(input_img, decoded)
vae.summary()

# TODO plot model

# TODO add custom loss

# Reconstruction loss
reconstruction_loss = binary_crossentropy(K.flatten(input_img), K.flatten(decoded))
reconstruction_loss *= img_size[0] * img_size[1] * img_size[2]
# KL Divergence
kl_loss = 1 + z_log_var - K.square(z_mu) - K.exp(z_log_var)
kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='Adam')
vae.fit(x_train, epochs=30, batch_size=256, shuffle=True, validation_split=0.1, verbose=2)

# generation result
decoded_img = vae.predict(x_test, workers=4)

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

