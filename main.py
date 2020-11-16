from unetpp import model as unetpp
from unet import model as unet
from tensorflow.keras.optimizers import Adam
from losses import dice, iou, weighted_loss, cce_iou_dice

weights_list = {1: 1.0, 2: 50.0, 3: 70.0}

if __name__ == '__main__':
    optimizer = Adam(lr=0.0005)

    unet.compile(optimizer=optimizer, loss=weighted_loss(cce_iou_dice, weights_list), metrics=[dice, iou])
    unet.summary()

    unetpp.compile(optimizer=optimizer, loss=weighted_loss(cce_iou_dice, weights_list), metrics=[dice, iou])
    unetpp.summary()
