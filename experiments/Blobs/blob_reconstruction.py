import sys
sys.path.append('/home/chen/MScProject/Code')

import torch
import matplotlib.pyplot as plt
from utils import torch_to_numpy
from splatCoords import splat_coord

for size in [64]:

    x_inputs = torch.tensor([[0.5,0.5,4,3,torch.pi/2]])    

    d = splat_coord(x_inputs, size)
    d = d.sum(0)

    plt.figure()
    plt.imshow(torch_to_numpy(d),vmin=0,vmax=1)
    plt.colorbar()

    plt.savefig(f'test_{size}.png')
    plt.close()


