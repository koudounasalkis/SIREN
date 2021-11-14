from torch.utils.data import Dataset
from PIL import Image
import skimage
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import scipy.ndimage

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    # img = img.convert('1') # grayscaling, by default to a bi-level colored image

    # Preprocessing functions     
    transform = Compose([
        Resize(sidelength),
        # Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) # image = (image - mean) / std, output values in [-1., +1.]
    ])
    img = transform(img)
    return img

def get_starfish_tensor(sidelength):
    img = Image.open('BSDS500/BSDS500/data/images/train/12003.jpg')
    
    # Adapt the original image
    size=[321, 321]
    img = img.crop((0, 0, size[0], size[1]))
    img = img.convert('L') # grayscaling, by default to a bi-level colored image

    # Preprocessing functions          
    transform = Compose([
        Resize(sidelength),
        # Grayscale(num_output_channels=1), # grayscaling
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class PoissonEqn(Dataset):
    def __init__(self, sidelength, img_type='cameraman', gradients=True, laplace=True):
        super().__init__()
        self.values = {}

        if img_type == 'cameraman':
            img = get_cameraman_tensor(sidelength)
        elif img_type == 'starfish':
            img = get_starfish_tensor(sidelength) # values in [-1., +1.]
        
        # Store image
        image = img.permute(1, 2, 0).view(-1, 1)
        self.values['img'] = image
        # Store gradients
        if gradients:
            grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
            grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
            self.values['grads'] = grads
        # Store laplacian
        if laplace:
            lapl = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
            lapl = torch.from_numpy(lapl).view(-1, 1)
            self.values['laplace'] = lapl
        
        self.image = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.values
