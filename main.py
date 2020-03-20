from model.unetpp import UNetPP

if __name__ == '__main__':
    unet = UNetPP()
    unet.compile()
    unet.summary()
