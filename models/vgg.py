import torch
import torch.nn as nn

class vgg_layer(nn.Module):
    def __init__(self, nin, nout, activation='l_relu'):
        super(vgg_layer, self).__init__()

        activations = {
           'l_relu': nn.LeakyReLU(0.2, inplace=True),
           'tanh': nn.Tanh()
        }

        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                activations[activation]
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, dim, nc=1, batch_size=None,activation='l_relu'):
        super(Encoder, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.nc = nc
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64, activation),
                vgg_layer(64, 64, activation),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128, activation),
                vgg_layer(128, 128, activation),
                )
        # 16 x 16
        self.c3 = nn.Sequential(
                vgg_layer(128, 256, activation),
                vgg_layer(256, 256, activation),
                vgg_layer(256, 256, activation),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512, activation),
                vgg_layer(512, 512, activation),
                vgg_layer(512, 512, activation),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


    def batch_seq_to_batch(self, x, size):
      return x.view(-1,self.nc,size,size)

    def batch_to_batch_seq(self, x, size):
      return x.view(self.batch_size,-1,x.size(-3),x.size(-2),x.size(-1))

    def forward(self, input):

        was_batch_seq = (input.dim() == 5)

        if was_batch_seq:
          input = self.batch_seq_to_batch(input,64)

        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 1
        x = h5.view(-1, self.dim)

        if was_batch_seq:
          h1 = self.batch_to_batch_seq(h1,64)
          h2 = self.batch_to_batch_seq(h2,32)
          h3 = self.batch_to_batch_seq(h3,16)
          h4 = self.batch_to_batch_seq(h4,8)
          h5 = self.batch_to_batch_seq(h5,4)
          x = x.view(self.batch_size,-1,self.dim)

        return x, [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self, dim, nc=1, batch_size = None, activation = 'l_relu'):
        super(Decoder, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.nc = nc
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
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
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

class DecoderNoSkip(nn.Module):
    def __init__(self, dim, nc=1):
        super(DecoderNoSkip, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128, 128),
                vgg_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(up1) # 8 x 8
        up2 = self.up(d2) # 8 -> 16
        d3 = self.upc3(up2) # 16 x 16
        up3 = self.up(d3) # 8 -> 32
        d4 = self.upc4(up3) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(up4) # 64 x 64
        return output
    
class VGGEncoderDecoder(nn.Module):
        def __init__(self, dim, nc=1, batch_size = None, activation = 'l_relu'):
                super(VGGEncoderDecoder, self).__init__()
                self.encoder = Encoder(dim,nc,batch_size,activation)
                self.decoder = Decoder(dim,3,batch_size,activation)
        
        def forward(self,input):
            return self.decoder(self.encoder(input))
        
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
                vgg_layer(256*2 + self.added_feature_d, 256, activation),
                vgg_layer(256, 256, activation),
                vgg_layer(256, 128, activation)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2 + self.added_feature_d, 128, activation),                
                vgg_layer(128, 64, activation)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2 + self.added_feature_d, 64, activation),                                            
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




