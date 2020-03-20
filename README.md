# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

UNet++ is a new general purpose image segmentation architecture for more accurate image segmentation. UNet++ consists of U-Nets of varying depths whose decoders are densely connected at the same resolution via the redesigned skip pathways, which aim to address two key challenges of the U-Net: 1) unknown depth of the optimal architecture and 2) the unnecessarily restrictive design of skip connections.

[![License](http://img.shields.io/:license-mit-green.svg?style=flat-square)](http://badges.mit-license.org)
![Network architecture](https://github.com/CodeAndChoke/UnetPlusPlus/blob/master/images/figure.png)

### Usage

Clone the repository and import the model with:

```python
from model.unetpp import UNetPP

if __name__ == '__main__':
    unet = UNetPP()
    unet.compile()
    unet.summary()
```

### Class imbalance

This model was designed for multi-classes classification and therefore uses [softmax](https://en.wikipedia.org/wiki/Softmax_function). 
as final activation. In order to fight class imbalance, which is usually the case, the high-order function 
`model.losses.weighted_loss` can be used to decorate classical loss functions.