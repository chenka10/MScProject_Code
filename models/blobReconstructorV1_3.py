import torch
import torch.nn as nn
from vgg import Encoder,MultiSkipsDecoder
from models.lstm import lstm
from splatCoords import splat_coord

def to_homogeneous(points):
    # Create a tensor of ones with the same size as the input tensor
    ones = torch.ones_like(points[:, :1])  # Create a tensor of ones with the same size as the first column of points
    
    # Concatenate the tensor of ones with the original tensor along the appropriate dimension
    homogeneous_points = torch.cat((points, ones), dim=1)
    
    return homogeneous_points

def from_homogeneous(points):
    points = points/points[:,-1].unsqueeze(-1)
    points = points[:,:2]

    return points


class PositionEncoder(nn.Module):
    def __init__(self):
        super(PositionEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(3,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,5))
                               

    def forward(self, positions):
        return self.enc(positions)        

class BlobConfig:
    def __init__(self, start_x, start_y, start_s, a_range, side):
        self.start_x = start_x
        self.start_y = start_y
        self.start_s = start_s
        self.a_range = a_range
        self.side = side

class BlobReconstructorV1_3(nn.Module):
    def __init__(self, lstm, position_encoders):
        super(BlobReconstructorV1_3, self).__init__()

        self.lstm = lstm
        self.position_encoders = position_encoders

        self.vgg_encoder = Encoder(256, 3)
        self.vgg_decoder = MultiSkipsDecoder(256,1,3)

      

    def forward(self, x_tm1, positions):

        blobs_data = []        
        device = positions.device
        
        for i,blob_config in enumerate(self.blob_configs):
            
            if blob_config.side == 'right':
                curr_pos = positions[:,:3]
            else:
                curr_pos = positions[:,3:]
                
            blob_data = self.position_encoders[i](curr_pos*100)                   
        
            blob_data[:,:2] = torch.sigmoid(blob_data[:,:2]) + torch.tensor([blob_config.start_y,blob_config.start_x]).to(device)
            blob_data[:,2] += blob_config.start_s
            blob_data[:,3] = blob_config.a_range[0] +torch.sigmoid(blob_data[:,3])*(blob_config.a_range[1]-blob_config.a_range[0])
            blob_data[:,4] = torch.sigmoid(blob_data[:,4])*torch.pi     

            blobs_data.append(blob_data)            
        
        blobs_images_visualization = None    
        
        size = 64
            
        blobs_opacities = []    
        blobs_images = []            
        for i,blob_data in enumerate(blobs_data):
            curr_blob_data = blob_data.clone()
            curr_blob_data[:,2]/=(2**i)
            blobs_opacities.append(splat_coord(curr_blob_data,size))
            blob_img = self.blob_transform[i](blobs_opacities[-1]*self.blobs_f[i,:][None,:,None,None])
            blobs_images.append(blob_img)        

        if blobs_images_visualization is None:
            blobs_images_visualization = [bo.detach().cpu() for bo in blobs_opacities]
        
        x_t_tilde = self.decoder(torch.randn(x_tm1.size()).to(device))

        b_t = torch.zeros(x_tm1.size()).to(device)
        ones = torch.ones(x_tm1.size()).to(device)

        for i,blob_img in enumerate(blobs_images):
            x_t_tilde=torch.mul(x_t_tilde,(1-blobs_opacities[i]))
            x_t_tilde=torch.add(x_t_tilde,blob_img[:,:3,:,:]*(blobs_opacities[i]))

            b_t=torch.mul(b_t,(1-blobs_opacities[i]))
            b_t=torch.add(b_t,ones*(blobs_opacities[i]))

        x_tm1_tilde = self.background_unet(x_tm1)
        output_high = self.unet(torch.concat([b_t,x_tm1_tilde],1))

        return output_high, x_t_tilde, b_t
    
