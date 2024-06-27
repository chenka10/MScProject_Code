from Code.models.vgg import VGGEncoderDecoder
from Code.models.vgg128 import VGGEncoderDecoder128

def get_vgg_enc_dec(im_size, dim, nc=1, batch_size = None, activation = 'l_relu'):
    if im_size not in [64,128]:
        raise ValueError('im_size must be 64 or 128')
    
    if im_size==64:
        return VGGEncoderDecoder(dim,nc,batch_size,activation)
    
    if im_size==128:
        return VGGEncoderDecoder128(dim,nc,batch_size,activation)
    
    raise ValueError('No matching VGG found')
    
