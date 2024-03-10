import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class JigsawsFrameEncoder(nn.Module):
    def __init__(self, d_model,nhead, numlayers, output_size, dropout):
        super(JigsawsFrameEncoder, self).__init__()
        self.output_size = output_size
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = d_model,nhead = nhead, batch_first=True,dropout=dropout),numlayers)

        # latent mean and variance
        self.mean_layer = nn.Linear(d_model, output_size)
        self.logvar_layer = nn.Linear(d_model, output_size)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterization(mu,torch.exp(0.5*logvar))

        return z, mu, logvar

class JigsawsFrameDecoder(nn.Module):
    def __init__(self, d_model,nhead, numlayers, output_size,dropout,causal = True):
        super(JigsawsFrameDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model = d_model,nhead = nhead,  batch_first=True, dropout = dropout),numlayers)
        self.fc = nn.Linear(d_model,output_size*2)
        self.causal = causal

        # latent mean and variance
        self.mean_layer = nn.Linear(output_size*2, output_size)
        self.logvar_layer = nn.Linear(output_size*2, output_size)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def forward(self, tgt, memory):
        msk = generate_square_subsequent_mask(tgt.size(1))

        if self.causal:
          x = self.decoder(tgt,memory,tgt_is_causal=True,tgt_mask = msk)
        else:
          x = self.decoder(tgt,memory)

        x = self.fc(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        x = self.reparameterization(mu,torch.exp(0.5*logvar))

        return x,mu,logvar