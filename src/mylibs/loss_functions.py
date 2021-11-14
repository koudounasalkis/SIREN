import torch
import torchvision
import myutils
from myutils import gradient, laplace

def image_mse(model_output, coords, gt_image):
    image_loss = ((model_output - gt_image)**2).mean()
    return image_loss

def gradients_mse(model_output, coords, gt_gradients):
    # compute gradients on the model
    gradients = gradient(model_output, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss
 
def laplace_mse(model_output, coords, gt_laplace):
    laplacian = laplace(model_output, coords)
    # compare them with the ground truth
    laplace_loss = torch.mean((laplacian - gt_laplace)**2)
    return laplace_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, with_image_mse=True):
        super(VGGPerceptualLoss, self).__init__()

        self.with_image_mse = with_image_mse

        # VGG19:
        # 64, 64, => 4 features
        # 'M', 128, 128, => 5 features
        # 'M', 256, 256, 256, 256, => 1+2*4 = 9 features
        # 'M', 512, 512, 512, 512, => 9 features
        # 'M', 512, 512, 512, 512, => 9 features
        # 'M'
        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval()) # 64, 64
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval()) # + 'M', 128, 128
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:18].eval()) # + 'M', 256, 256, 256, 256
        blocks.append(torchvision.models.vgg19(pretrained=True).features[18:27].eval()) #  + 'M', 512, 512, 512, 512
        blocks.append(torchvision.models.vgg19(pretrained=True).features[27:36].eval()) #  + 'M', 512, 512, 512, 512
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3: # to "RGB"
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize: # adapt the input for the VGG net
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target

        if self.with_image_mse: # MSE on the image too, not only on the features
            loss += torch.nn.functional.mse_loss(x, y)
        for block in self.blocks:
            x = block(x)
            y = block(y)

            loss += torch.nn.functional.mse_loss(x, y)
        return loss