import tensorflow as tf
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose


batch_size = 2
epochs = 25
number_of_filters = 2
num_classes = 3
fil_coef = 2
leaky_alpha = 0.1
dropout_rate = 0.25


def conv2d(filters: int):
    return Conv2D(filters=filters,kernel_size=(3, 3),padding='same',kernel_regularizer=l2(0.),bias_regularizer=l2(0.))


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters,kernel_size=(2, 2),strides=(2, 2),padding='same')


model_input = Input((512, 512, 3))
x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
x00 = BatchNormalization()(x00)
x00 = LeakyReLU(0.01)(x00)
x00 = Dropout(0.2)(x00)
x00 = conv2d(filters=int(16 * number_of_filters))(x00)
x00 = BatchNormalization()(x00)
x00 = LeakyReLU(0.01)(x00)
x00 = Dropout(0.2)(x00)
p0 = MaxPooling2D(pool_size=(2, 2))(x00)

x10 = conv2d(filters=int(32 * number_of_filters))(p0)
x10 = BatchNormalization()(x10)
x10 = LeakyReLU(0.01)(x10)
x10 = Dropout(0.2)(x10)
x10 = conv2d(filters=int(32 * number_of_filters))(x10)
x10 = BatchNormalization()(x10)
x10 = LeakyReLU(0.01)(x10)
x10 = Dropout(0.2)(x10)
p1 = MaxPooling2D(pool_size=(2, 2))(x10)

x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
x01 = concatenate([x00, x01])
x01 = conv2d(filters=int(16 * number_of_filters))(x01)
x01 = BatchNormalization()(x01)
x01 = LeakyReLU(0.01)(x01)
x01 = conv2d(filters=int(16 * number_of_filters))(x01)
x01 = BatchNormalization()(x01)
x01 = LeakyReLU(0.01)(x01)
x01 = Dropout(0.2)(x01)

x20 = conv2d(filters=int(64 * number_of_filters))(p1)
x20 = BatchNormalization()(x20)
x20 = LeakyReLU(0.01)(x20)
x20 = Dropout(0.2)(x20)
x20 = conv2d(filters=int(64 * number_of_filters))(x20)
x20 = BatchNormalization()(x20)
x20 = LeakyReLU(0.01)(x20)
x20 = Dropout(0.2)(x20)
p2 = MaxPooling2D(pool_size=(2, 2))(x20)

x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
x11 = concatenate([x10, x11])
x11 = conv2d(filters=int(16 * number_of_filters))(x11)
x11 = BatchNormalization()(x11)
x11 = LeakyReLU(0.01)(x11)
x11 = conv2d(filters=int(16 * number_of_filters))(x11)
x11 = BatchNormalization()(x11)
x11 = LeakyReLU(0.01)(x11)
x11 = Dropout(0.2)(x11)

x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
x02 = concatenate([x00, x01, x02])
x02 = conv2d(filters=int(16 * number_of_filters))(x02)
x02 = BatchNormalization()(x02)
x02 = LeakyReLU(0.01)(x02)
x02 = conv2d(filters=int(16 * number_of_filters))(x02)
x02 = BatchNormalization()(x02)
x02 = LeakyReLU(0.01)(x02)
x02 = Dropout(0.2)(x02)

x30 = conv2d(filters=int(128 * number_of_filters))(p2)
x30 = BatchNormalization()(x30)
x30 = LeakyReLU(0.01)(x30)
x30 = Dropout(0.2)(x30)
x30 = conv2d(filters=int(128 * number_of_filters))(x30)
x30 = BatchNormalization()(x30)
x30 = LeakyReLU(0.01)(x30)
x30 = Dropout(0.2)(x30)
p3 = MaxPooling2D(pool_size=(2, 2))(x30)

x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
x21 = concatenate([x20, x21])
x21 = conv2d(filters=int(16 * number_of_filters))(x21)
x21 = BatchNormalization()(x21)
x21 = LeakyReLU(0.01)(x21)
x21 = conv2d(filters=int(16 * number_of_filters))(x21)
x21 = BatchNormalization()(x21)
x21 = LeakyReLU(0.01)(x21)
x21 = Dropout(0.2)(x21)

x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
x12 = concatenate([x10, x11, x12])
x12 = conv2d(filters=int(16 * number_of_filters))(x12)
x12 = BatchNormalization()(x12)
x12 = LeakyReLU(0.01)(x12)
x12 = conv2d(filters=int(16 * number_of_filters))(x12)
x12 = BatchNormalization()(x12)
x12 = LeakyReLU(0.01)(x12)
x12 = Dropout(0.2)(x12)

x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
x03 = concatenate([x00, x01, x02, x03])
x03 = conv2d(filters=int(16 * number_of_filters))(x03)
x03 = BatchNormalization()(x03)
x03 = LeakyReLU(0.01)(x03)
x03 = conv2d(filters=int(16 * number_of_filters))(x03)
x03 = BatchNormalization()(x03)
x03 = LeakyReLU(0.01)(x03)
x03 = Dropout(0.2)(x03)

m = conv2d(filters=int(256 * number_of_filters))(p3)
m = BatchNormalization()(m)
m = LeakyReLU(0.01)(m)
m = conv2d(filters=int(256 * number_of_filters))(m)
m = BatchNormalization()(m)
m = LeakyReLU(0.01)(m)
m = Dropout(0.2)(m)

x31 = conv2dtranspose(int(128 * number_of_filters))(m)
x31 = concatenate([x31, x30])
x31 = conv2d(filters=int(128 * number_of_filters))(x31)
x31 = BatchNormalization()(x31)
x31 = LeakyReLU(0.01)(x31)
x31 = conv2d(filters=int(128 * number_of_filters))(x31)
x31 = BatchNormalization()(x31)
x31 = LeakyReLU(0.01)(x31)
x31 = Dropout(0.2)(x31)

x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
x22 = concatenate([x22, x20, x21])
x22 = conv2d(filters=int(64 * number_of_filters))(x22)
x22 = BatchNormalization()(x22)
x22 = LeakyReLU(0.01)(x22)
x22 = conv2d(filters=int(64 * number_of_filters))(x22)
x22 = BatchNormalization()(x22)
x22 = LeakyReLU(0.01)(x22)
x22 = Dropout(0.2)(x22)

x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
x13 = concatenate([x13, x10, x11, x12])
x13 = conv2d(filters=int(32 * number_of_filters))(x13)
x13 = BatchNormalization()(x13)
x13 = LeakyReLU(0.01)(x13)
x13 = conv2d(filters=int(32 * number_of_filters))(x13)
x13 = BatchNormalization()(x13)
x13 = LeakyReLU(0.01)(x13)
x13 = Dropout(0.2)(x13)

x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
x04 = conv2d(filters=int(16 * number_of_filters))(x04)
x04 = BatchNormalization()(x04)
x04 = LeakyReLU(0.01)(x04)
x04 = conv2d(filters=int(16 * number_of_filters))(x04)
x04 = BatchNormalization()(x04)
x04 = LeakyReLU(0.01)(x04)
x04 = Dropout(0.2)(x04)
output = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(x04)
model = tf.keras.Model(inputs=[model_input], outputs=[output])
