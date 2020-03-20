import tensorflow as tf
import tensorflow.keras.backend as K
import typing


def weighted_loss(original_loss_function: typing.Callable, weights_list: dict) -> typing.Callable:
    def loss_function(true, pred):
        class_selectors = tf.cast(K.argmax(true, axis=-1), tf.int32)
        class_selectors = [K.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_function(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_function


@tf.function
def cce_iou_dice(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) \
           + iou_weight * log_iou(y_true, y_pred, smooth) \
           + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def cce_iou(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) + iou_weight * log_iou(y_true, y_pred, smooth)


@tf.function
def cce_dice(y_true, y_pred, smooth=1, cat_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def bce_iou(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1):
    return cat_weight * K.binary_crossentropy(y_true, y_pred) + iou_weight * log_iou(y_true, y_pred, smooth)


@tf.function
def bce_dice(y_true, y_pred, smooth=1, cat_weight=1, dice_weight=1):
    return cat_weight * K.binary_crossentropy(y_true, y_pred) + dice_weight * log_dice(y_true, y_pred, smooth)


@tf.function
def log_iou(y_true, y_pred, smooth=1):
    return - K.log(iou(y_true, y_pred, smooth))


@tf.function
def log_dice(y_true, y_pred, smooth=1):
    return -K.log(dice(y_true, y_pred, smooth))


@tf.function
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
