import sys
sys.path.append('/home/chen/MScProject/')
sys.path.append('/home/chen/MScProject/Code')

import torch
import matplotlib.pyplot as plt
from utils import torch_to_numpy
from Code.splatCoords import splat_coord


def rotate_point(point, pivot, theta):
    """
    Rotate a point around a pivot by theta degrees.

    :param point: Tuple (x, y) representing the point to rotate.
    :param pivot: Tuple (x, y) representing the pivot point around which to rotate.
    :param theta: The angle in degrees by which to rotate the point.
    :return: Tuple (x', y') representing the rotated point.
    """
    # Convert theta to radians
    theta_rad = -theta
    
    # Translate point to origin
    translated_x = point[0] - pivot[0]
    translated_y = point[1] - pivot[1]
    
    # Apply the rotation matrix
    rotated_x = translated_x * torch.cos(theta_rad) - translated_y * torch.sin(theta_rad)
    rotated_y = translated_x * torch.sin(theta_rad) + translated_y * torch.cos(theta_rad)
    
    # Translate the point back
    final_x = rotated_x + pivot[0]
    final_y = rotated_y + pivot[1]
    
    return (final_x.item(), final_y.item())

for size in [64]:

    x_1 = 0.5
    y_1 = 0.5
    s_1 = 10
    a_1 = 2
    theta_1 = torch.pi/4

    y_2 = y_1- 0.1*a_1 -0.007*s_1 
    x_2 = x_1

    x_2,y_2 = rotate_point(torch.tensor([x_2,y_2]),torch.tensor([x_1,y_1]),torch.tensor(theta_1))

    x_inputs = torch.tensor([[y_1,x_1,s_1,a_1,theta_1],
                             [y_2,x_2,1,3,torch.pi]])    

    d = splat_coord(x_inputs, size)
    d = d.sum(0)

    plt.figure()
    plt.imshow(torch_to_numpy(d),vmin=0,vmax=1)
    plt.colorbar()

    plt.savefig(f'test2_{size}.png')
    plt.close()


