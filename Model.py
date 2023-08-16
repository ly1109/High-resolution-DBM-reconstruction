import tensorflow as tf
from keras.layers import Input, Conv2D, Concatenate, Add, GlobalMaxPooling2D, Reshape, Dense, Multiply, UpSampling2D, \
    Resizing, AveragePooling2D, Conv2DTranspose, Dropout, GlobalMaxPool2D, GlobalAveragePooling2D, Activation, SpatialDropout2D
from keras.models import Model

ROWS = 8
COLS = 8
num_filter = 64
p = 0.1
scale = 15


def dcscn():
    input3 = Input(shape=(ROWS * 4, COLS * 4, 1), name='input3')
    conv1 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(input3)
    conv2 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv1)
    conv3 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv2)
    conv4 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv3)
    conv5 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv4)
    conv6 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv5)
    conv7 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv6)
    conv8 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv7)
    conv9 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv8)
    conv10 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv9)
    conv11 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv10)
    conv12 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv11)

    concat1 = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12])
    concat1 = Dropout(p)(concat1)

    conv13 = Conv2D(num_filter, 1, activation=tf.nn.leaky_relu, padding='same')(concat1)

    conv14 = Conv2D(num_filter, 1, activation=tf.nn.leaky_relu, padding='same')(concat1)
    conv14 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv14)

    concat2 = Concatenate()([conv13, conv14])

    conv15 = Conv2D(scale * scale, 3, activation=tf.nn.leaky_relu, padding='same')(concat2)

    pix16 = tf.nn.depth_to_space(conv15, scale)

    up17 = UpSampling2D(size=(scale, scale), interpolation='bicubic')(input3)

    add1 = Add()([up17, pix16])

    model = Model(inputs=input3, outputs=add1)

    return model


def dcscn_dbm():
    input3 = Input(shape=(ROWS * 4, COLS * 4, 1), name='input3')

    conv0 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(input3)

    conv1 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv0)
    conv2 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv1)
    add1 = Add()([conv0, conv2])
    conv3 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(add1)
    conv4 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv3)
    add2 = Add()([add1, conv4])
    conv5 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(add2)
    conv6 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv5)
    add3 = Add()([add2, conv6])
    conv7 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(add3)
    conv8 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv7)
    add4 = Add()([add3, conv8])
    conv9 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(add4)
    conv10 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv9)
    add5 = Add()([add4, conv10])
    conv11 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(add5)
    conv12 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv11)
    add6 = Add()([add5, conv12])

    concat1 = Concatenate()([add1, add2, add3, add4, add5, add6])
    concat1 = Dropout(p)(concat1)

    conv13 = Conv2D(num_filter, 1, activation=tf.nn.leaky_relu, padding='same')(concat1)

    conv14 = Conv2D(num_filter, 1, activation=tf.nn.leaky_relu, padding='same')(concat1)
    conv14 = Conv2D(num_filter, 3, activation=tf.nn.leaky_relu, padding='same')(conv14)

    concat2 = Concatenate()([conv13, conv14])

    conv15 = Conv2D(scale * scale, 3, activation=tf.nn.leaky_relu, padding='same')(concat2)

    pix16 = tf.nn.depth_to_space(conv15, scale)

    up17 = UpSampling2D(size=(scale, scale), interpolation='bicubic')(input3)

    add = Add()([up17, pix16])

    # conv17 = Conv2D(1, 1, activation='linear')(add1)

    model = Model(inputs=input3, outputs=add)

    return model
