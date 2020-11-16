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
c0 = conv2d(filters=int(16 * fil_coef))(model_input)
c0 = BatchNormalization()(c0)
c0 = LeakyReLU(leaky_alpha)(c0)
c0 = Dropout(dropout_rate)(c0)
c0 = conv2d(filters=int(16 * fil_coef))(c0)
c0 = BatchNormalization()(c0)
c0 = LeakyReLU(leaky_alpha)(c0)
c0 = Dropout(dropout_rate)(c0)
p0 = MaxPooling2D(pool_size=(2, 2))(c0)

c1 = conv2d(filters=int(32 * fil_coef))(p0)
c1 = BatchNormalization()(c1)
c1 = LeakyReLU(leaky_alpha)(c1)
c1 = Dropout(dropout_rate)(c1)
c1 = conv2d(filters=int(32 * fil_coef))(c1)
c1 = BatchNormalization()(c1)
c1 = LeakyReLU(leaky_alpha)(c1)
c1 = Dropout(dropout_rate)(c1)
p1 = MaxPooling2D(pool_size=(2, 2))(c1)

c2 = conv2d(filters=int(64 * fil_coef))(p1)
c2 = BatchNormalization()(c2)
c2 = LeakyReLU(leaky_alpha)(c2)
c2 = Dropout(dropout_rate)(c2)
c2 = conv2d(filters=int(64 * fil_coef))(c2)
c2 = BatchNormalization()(c2)
c2 = LeakyReLU(leaky_alpha)(c2)
c2 = Dropout(dropout_rate)(c2)
p2 = MaxPooling2D(pool_size=(2, 2))(c2)

c3 = conv2d(filters=int(128 * fil_coef))(p2)
c3 = BatchNormalization()(c3)
c3 = LeakyReLU(leaky_alpha)(c3)
c3 = Dropout(dropout_rate)(c3)
c3 = conv2d(filters=int(128 * fil_coef))(c3)
c3 = BatchNormalization()(c3)
c3 = LeakyReLU(leaky_alpha)(c3)
c3 = Dropout(dropout_rate)(c3)
p3 = MaxPooling2D(pool_size=(2, 2))(c3)

m = conv2d(filters=int(256 * fil_coef))(p3)
m = BatchNormalization()(m)
m = LeakyReLU(leaky_alpha)(m)
m = conv2d(filters=int(256 * fil_coef))(m)
m = BatchNormalization()(m)
m = LeakyReLU(leaky_alpha)(m)
m = Dropout(dropout_rate)(m)

u5 = conv2dtranspose(int(128 * fil_coef))(m)
u5 = concatenate([u5, c3])
c5 = conv2d(filters=int(128 * fil_coef))(u5)
c5 = BatchNormalization()(c5)
c5 = LeakyReLU(leaky_alpha)(c5)
c5 = conv2d(filters=int(128 * fil_coef))(c5)
c5 = BatchNormalization()(c5)
c5 = LeakyReLU(leaky_alpha)(c5)
c5 = Dropout(dropout_rate)(c5)

u6 = conv2dtranspose(int(64 * fil_coef))(c5)
u6 = concatenate([u6, c2])
c6 = conv2d(filters=int(64 * fil_coef))(u6)
c6 = BatchNormalization()(c6)
c6 = LeakyReLU(leaky_alpha)(c6)
c6 = conv2d(filters=int(64 * fil_coef))(c6)
c6 = BatchNormalization()(c6)
c6 = LeakyReLU(leaky_alpha)(c6)
c6 = Dropout(dropout_rate)(c6)

u7 = conv2dtranspose(int(32 * fil_coef))(c6)
u7 = concatenate([u7, c1])
c7 = conv2d(filters=int(32 * fil_coef))(u7)
c7 = BatchNormalization()(c7)
c7 = LeakyReLU(leaky_alpha)(c7)
c7 = conv2d(filters=int(32 * fil_coef))(c7)
c7 = BatchNormalization()(c7)
c7 = LeakyReLU(leaky_alpha)(c7)
c7 = Dropout(dropout_rate)(c7)

u8 = conv2dtranspose(int(16 * fil_coef))(c7)
u8 = concatenate([u8, c0], axis=3)
c8 = conv2d(filters=int(16 * fil_coef))(u8)
c8 = BatchNormalization()(c8)
c8 = LeakyReLU(leaky_alpha)(c8)
c8 = conv2d(filters=int(16 * fil_coef))(c8)
c8 = BatchNormalization()(c8)
c8 = LeakyReLU(leaky_alpha)(c8)
c8 = Dropout(dropout_rate)(c8)

output = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(c8)
model = tf.keras.Model(inputs=[model_input], outputs=[output])