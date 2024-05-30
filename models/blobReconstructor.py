import torch
import torch.nn as nn
from vgg import vgg_layer, Encoder, VGGEncoderDecoder
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


class MultiSkipsDecoder(nn.Module):
    def __init__(self, dim, added_feature_d, nc=1, batch_size = None, activation = 'l_relu'):
        super(MultiSkipsDecoder, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.nc = nc   
        self.added_feature_d = added_feature_d

        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512, activation),
                vgg_layer(512, 512, activation),
                vgg_layer(512, 256, activation)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256, activation),
                vgg_layer(256, 256, activation),
                vgg_layer(256, 128, activation)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128, activation),                
                vgg_layer(128, 64, activation)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64, activation),                                            
                nn.Conv2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input

        was_batch_seq = (vec.dim() == 3)

        if was_batch_seq:
          skip = [inskip.view(-1,inskip.size(-3),inskip.size(-2),inskip.size(-1)) for inskip in skip]

        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16
        up3 = self.up(d3) # 8 -> 32
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64

        if was_batch_seq:
          output = output.view(self.batch_size,-1,self.nc,64,64)

        return output

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

class PositionToBlobs(nn.Module):
    def __init__(self, blob_configs):
        super(PositionToBlobs, self).__init__()

        self.blob_configs = blob_configs        
        self.position_encoders = nn.ModuleList()

        for blob_config in blob_configs:
            self.position_encoders.append(PositionEncoder())

    def forward(self, positions):

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

        return blobs_data
    
class BlobsToFeatureMaps(nn.Module):
    def __init__(self, blob_f_size, target_image_size):
        super(BlobsToFeatureMaps, self).__init__()
        self.blobs_f = nn.Parameter(torch.randn(blob_f_size))
        self.target_image_size = target_image_size
        self.blob_transform = nn.Sequential(
            vgg_layer(blob_f_size,64,'l_relu'),
            vgg_layer(64,64,'l_relu'),
            vgg_layer(64,blob_f_size,'l_relu'),
        )

    def forward(self, blob_data):
        size = self.target_image_size            
        curr_blob_data = blob_data.clone()            
        blobs_grayscale_map = splat_coord(curr_blob_data,size)
        blobs_feature_map = self.blob_transform(blobs_grayscale_map*self.blobs_f[None,:,None,None])

        return blobs_feature_map, blobs_grayscale_map


class BlobReconstructor(nn.Module):
    def __init__(self, hidden_dim, blob_configs, batch_size = None, activation = 'l_relu'):
        super(BlobReconstructor, self).__init__()
        self.encoder_background = Encoder(hidden_dim, 3, batch_size, activation)        

        blob_f_size = 4

        self.positions_to_blobs = PositionToBlobs(blob_configs)
        self.blobs_to_maps = nn.ModuleList()

        for _ in blob_configs:
            self.blobs_to_maps.append(BlobsToFeatureMaps(blob_f_size,64))
        
        self.blob_configs = blob_configs
        self.num_blobs = len(blob_configs)        

        self.blobs_f = nn.Parameter(torch.randn(self.num_blobs,blob_f_size))        
        self.decoder = VGGEncoderDecoder(hidden_dim,3,batch_size,activation)

        self.unet = VGGEncoderDecoder(64,3,batch_size,activation)          

    def forward(self, backgrounds, positions):  

        blobs_data = self.positions_to_blobs(positions)          
        
        blobs_images_visualization = None    

        blobs_feature_maps = []
        blobs_grayscale_maps = []

        for blob_i in range(len(blobs_data)):
            blobs_feature_map, blobs_grayscale_map = self.blobs_to_maps[blob_i](blobs_data[blob_i])     
            blobs_feature_maps.append(blobs_feature_map)
            blobs_grayscale_maps.append(blobs_grayscale_map)     

        if blobs_images_visualization is None:
            blobs_images_visualization = [bo.detach().cpu() for bo in blobs_grayscale_maps]
        
        output_low = self.decoder(backgrounds)        

        for i,blob_img in enumerate(blobs_feature_maps):
            output_low=torch.mul(output_low,(1-blobs_grayscale_maps[i]))
            output_low=torch.add(output_low,blob_img[:,:3,:,:]*(blobs_grayscale_maps[i]))

        output_high = self.unet(output_low)

        return output_high, output_low, blobs_grayscale_maps
    
