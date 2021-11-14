# SISR X16

## Final results

> WARNING: SRGAN results are wrong for the X16 test, please do not consider them. Even for the images and the graphs.

### Image 143

```
SISR 32x32 to 512x512
hidden layers: 3
hidden features: 256
total steps: 1000


PSNR:
	ReLU: 12.785148
	ReLU VGG: 12.595752
	ReLU NeRF: 13.322075
	ReLU NeRF VGG: 13.342308


SSIM:
	ReLU: 0.166977
	ReLU VGG: 0.173914
	ReLU NeRF: 0.170410
	ReLU NeRF VGG: 0.171085
```

```
SISR 32x32 to 512x512
hidden layers: 3
hidden features: 256
total steps: 1000


PSNR:
	SIREN: 13.510377
	SIREN VGG: 13.514738
	SIREN NeRF: 13.401980
	SIREN NeRF VGG: 13.348608


SSIM:
	SIREN: 0.187580
	SIREN VGG: 0.187872
	SIREN NeRF: 0.180577
	SIREN NeRF VGG: 0.171389
```