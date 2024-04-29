import torch
import torch.nn as nn
from vgg import vgg_layer, Encoder
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
                vgg_layer(64*2+self.added_feature_d, 64, activation),
                vgg_layer(64, 32, activation),
                vgg_layer(32, 16, activation),
                nn.ConvTranspose2d(16, nc, 3, 1, 1),
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
        self.enc1 = nn.Sequential(nn.Linear(3,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,5))
        
        self.enc2 = nn.Sequential(nn.Linear(3,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,128),
                               nn.LeakyReLU(),
                               nn.Linear(128,5))
                               

    def forward(self, positions):
        res_right = self.enc1(positions[:,:3])
        res_left = self.enc2(positions[:,3:])

        return torch.concat([res_right,res_left],dim=-1)


class BlobReconstructor(nn.Module):
    def __init__(self, hidden_dim, d=128, batch_size = None, activation = 'l_relu'):
        super(BlobReconstructor, self).__init__()
        self.encoder_background = Encoder(hidden_dim, 3, batch_size, activation)
        self.encoder_blobs = Encoder(hidden_dim, 1, batch_size, activation)     
        self.decoder = MultiSkipsDecoder(hidden_dim,d, 3, batch_size, activation)  
        self.position_encoder = PositionEncoder() 
        self.d = d

    def forward(self, backgrounds, positions):
        
        blob_data = self.position_encoder(positions)
        device = blob_data.device
        
        blob_data[:,:2] = torch.sigmoid(blob_data[:,:2]) + torch.tensor([0,0.25]).to(blob_data.device)
        blob_data[:,2]+=4
        blob_data[:,3] = 1+torch.sigmoid(blob_data[:,3])*2
        blob_data[:,4] = torch.sigmoid(blob_data[:,4])*torch.pi                

        n_blob = 5
        blob_data[:,n_blob:(2+n_blob)] = torch.sigmoid(blob_data[:,n_blob:(2+n_blob)]) + torch.tensor([0,-0.25]).to(blob_data.device)
        blob_data[:,(2+n_blob)]+=4
        blob_data[:,(3+n_blob)] = 1+torch.sigmoid(blob_data[:,(3+n_blob)])*2
        blob_data[:,(4+n_blob)] = torch.sigmoid(blob_data[:,(4+n_blob)])*torch.pi        

        vec_background,skips_background = self.encoder_background(backgrounds)

        backgrounds_signal = torch.tensor(list(range(self.d))).to(device)%3
        right_signal = torch.tensor(list(range(1,self.d+1))).to(device)%3
        left_signal = torch.tensor(list(range(2,self.d+2))).to(device)%3

        concat_skips = []
        first_blobs_1 = None
        first_blobs_2 = None
        for skip in skips_background:
            size = skip.shape[-1]            
            if size>=64:                
                opacity_right = splat_coord(blob_data[:,:5],size)
                opacity_left = splat_coord(blob_data[:,5:],size)   
                opacity_background = torch.ones_like(opacity_right)

                if first_blobs_1 is None:
                    first_blobs_1 = opacity_right.detach().cpu()
                    first_blobs_2 = opacity_left.detach().cpu()   

                alpha_background = opacity_background*(1-opacity_left)*(1-opacity_right)
                alpha_left = opacity_left*(1-opacity_right)
                alpha_right = opacity_right

                feature_map = alpha_background*backgrounds_signal[None,:,None,None] + alpha_right*right_signal[None,:,None,None] + alpha_left*left_signal[None,:,None,None]
                concat_skips.append(torch.concat([skip, feature_map],dim=1))
            else:
                concat_skips.append(skip)
        
        concat = [vec_background, concat_skips]
        output = self.decoder(concat)

        return output, first_blobs_1, first_blobs_2
    
