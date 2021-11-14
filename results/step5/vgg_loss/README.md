# VGG perceptual loss

## Configuration

SISR task for 224x224 to 512x512 on Cameraman image.

SIREN with [2, 3] hidden layers, 256 features.

Training on 10,000 steps, LR = 1e-4, with ADAM optimizer.

Default loss functions image MSE, compared with vgg perceptual loss obtained from a VGG19 pretrained.

## VGG loss effect

The VGG perceptual loss boosts the image fitting on the initial training steps. See images trained on 500 steps.

With many steps (10,000 or more) this effect is no more perceptible.