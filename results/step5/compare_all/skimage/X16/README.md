# SISR X16

## Final results

> WARNING: SRGAN results are wrong for the X16 test, please do not consider them. Even for the images and the graphs.

### Image cameraman

```
SISR 32x32 to 512x512
hidden layers: 3
hidden features: 512
total steps: 500


PSNR:
	SIREN: 19.924537
	SIREN VGG: 19.957822
	SIREN Nerf: 19.596876
	SIREN Nerf VGG: 19.797087
	ReLU Nerf: 19.494541
	bicubic: 20.273425
	SRGAN: 24.672834


SSIM:
	SIREN: 0.629094
	SIREN VGG: 0.631650
	SIREN Nerf: 0.604623
	SIREN Nerf VGG: 0.618829
	ReLU Nerf: 0.564109
	bicubic: 0.629328
	SRGAN: 0.770400
```