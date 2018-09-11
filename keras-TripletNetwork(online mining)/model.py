from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten, ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout, GlobalMaxPooling2D
from keras.applications.mobilenet import MobileNet


def build_mnist_model(input_shape=(28, 28, 1)):
    def _conv_block(filters, kernel_size, name):
        def _layer(x):
            x = Conv2D(filters, kernel_size=kernel_size, padding='same', name='conv_' + name)(x)
            x = ReLU()(x)
            x = MaxPooling2D()(x)
            return x

        return _layer

    input = Input(shape=input_shape, name='input')
    x = input
    num_channels = 3
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        x = _conv_block(c, kernel_size=3, name=str(i))(x)

    x = Flatten()(x)
    x = Dense(64)(x)

    return Model(inputs=input, outputs=x)


def build_cifar10_model(input_shape=(32, 32, 3)):
    def _conv_block(filters, kernel_size, name):
        def _layer(x):
            x = Conv2D(filters, kernel_size=kernel_size, padding='same', name='conv_' + name)(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = LeakyReLU(alpha=0.3)(x)
            # x = Dropout(0.25)(x)
            return x

        return _layer

    input = Input(shape=input_shape, name='input')
    x = input

    x = _conv_block(64, kernel_size=5, name=str(0))(x)
    x = _conv_block(128, kernel_size=3, name=str(1))(x)
    x = _conv_block(256, kernel_size=3, name=str(2))(x)

    # 1 Final Conv to get into 128 dim embedding
    x = Conv2D(128, kernel_size=2, padding='same')(x)
    x = GlobalMaxPooling2D()(x)

    # base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    # x = base_model.output
    # x = Flatten()(x)
    # x = Dense(128)(x)
    #
    # model = Model(base_model.input, x)
    #
    # return model
    return Model(inputs=input, outputs=x)
