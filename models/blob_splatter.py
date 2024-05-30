import torch
import torch.nn as nn
from blob_position_encoder import PositionEncoder
from Code.splatCoords import splat_coord

class BlobConfig:
    def __init__(self, start_x, start_y, start_s, a_range, side):
        self.start_x = start_x
        self.start_y = start_y
        self.start_s = start_s
        self.a_range = a_range
        self.side = side

class BlobSplatter(nn.Module):
    """
    This module taks an array of positions (batch_size,6),
    and return an arrays of splatted blob images according to specified "result_dims" (which may look like [16,32,64]).

    Returned is a list of length (len(results_dims)),
    every element (i) in the list is of size (batch_size,results_dims[i],results_dims[i]) 
    """
    def __init__(self, blob_configs, result_dims, position_encoders=None):
        super(BlobSplatter, self).__init__()

        self.blob_configs = blob_configs
        self.result_dims = result_dims

        if position_encoders is None:
            self.position_encoders = nn.ModuleList()
            for blob_config in blob_configs:
                self.position_encoders.append(PositionEncoder())        
        else:
            self.position_encoders = position_encoders

                               
    def forward(self, positions, target_size, blobs_scale_factor):
        """
        Inputs:
        positions [torch.tensor(batch_size,6)] - the positions to use for blob generation
        target_size [number] - the size of the blob map to generate (i.e. if target_size=64 the maps wil be 64x64)
        blobs_scale_factor [number] - a scale factor for the resulting blobs (usefull if position encoders where trained on images with different size from the one you try to generate)
        """

        batch_size = positions.size(0)
        
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

        blobs_images = torch.zeros([batch_size,target_size,target_size])

        for i,blob_data in enumerate(blobs_data):
            curr_blob_data = blob_data.clone()
            curr_blob_data[:,2]*=blobs_scale_factor
            curr_blob_image = splat_coord(curr_blob_data,target_size)

            blobs_images = torch.mul(blobs_images,curr_blob_image)
            blobs_images = torch.add(blobs_images,curr_blob_image)

        return blobs_images

            
            
            