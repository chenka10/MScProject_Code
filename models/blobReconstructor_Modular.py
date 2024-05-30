import torch
import torch.nn as nn
from vgg import vgg_layer, Encoder, VGGEncoderDecoder
from blob_splatter import BlobSplatter

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

class BlobReconstructor(nn.Module):
    def __init__(self, hidden_dim, blob_configs, batch_size = None, activation = 'l_relu'):
        super(BlobReconstructor, self).__init__()
        self.encoder_background = Encoder(hidden_dim, 3, batch_size, activation)        
        self.blobs_splatter = BlobSplatter(blob_config,64)
        self.blob_transform = nn.ModuleList()  
        for blob_config in blob_configs:            
            self.blob_transform.append(nn.Sequential(
                vgg_layer(4,64,activation),
                vgg_layer(64,64,activation),
                vgg_layer(64,4,activation),
            ))           
        
        self.blob_configs = blob_configs
        self.num_blobs = len(blob_configs)

        blob_f_size = 4

        self.blobs_f = nn.Parameter(torch.randn(self.num_blobs,blob_f_size))        
        self.decoder = VGGEncoderDecoder(hidden_dim,3,batch_size,activation)

        self.unet = VGGEncoderDecoder(64,3,batch_size,activation)          

    def forward(self, backgrounds, positions):

        blobs_data = []        
        device = positions.device
        
        # for i,blob_config in enumerate(self.blob_configs):
            
        #     if blob_config.side == 'right':
        #         curr_pos = positions[:,:3]
        #     else:
        #         curr_pos = positions[:,3:]
                
        #     blob_data = self.position_encoders[i](curr_pos*100)                   
        
        #     blob_data[:,:2] = torch.sigmoid(blob_data[:,:2]) + torch.tensor([blob_config.start_y,blob_config.start_x]).to(device)
        #     blob_data[:,2] += blob_config.start_s
        #     blob_data[:,3] = blob_config.a_range[0] +torch.sigmoid(blob_data[:,3])*(blob_config.a_range[1]-blob_config.a_range[0])
        #     blob_data[:,4] = torch.sigmoid(blob_data[:,4])*torch.pi     

        #     blobs_data.append(blob_data)            
        
        # blobs_images_visualization = None    
        
        # size = 64
            
        # blobs_opacities = []    
        # blobs_images = []            
        # for i,blob_data in enumerate(blobs_data):
        #     curr_blob_data = blob_data.clone()
        #     curr_blob_data[:,2]/=(2**i)
        #     blobs_opacities.append(splat_coord(curr_blob_data,size))
        #     blob_img = self.blob_transform[i](blobs_opacities[-1]*self.blobs_f[i,:][None,:,None,None])
        #     blobs_images.append(blob_img)     

        blobs_   

        if blobs_images_visualization is None:
            blobs_images_visualization = [bo.detach().cpu() for bo in blobs_opacities]
        
        output_low = self.decoder(backgrounds)        

        for i,blob_img in enumerate(blobs_images):
            output_low=torch.mul(output_low,(1-blobs_opacities[i]))
            output_low=torch.add(output_low,blob_img[:,:3,:,:]*(blobs_opacities[i]))

        output_high = self.unet(output_low)

        return output_high, output_low, blobs_opacities
    
