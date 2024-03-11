import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')
sys.path.append('/home/chen/MScProject/Code/experiments/Transformer_AutoEncoder')

from JigsawsKinematicsDataset import JigsawsKinematicsDataset
from JigsawsImageDataset import JigsawsImageDataset
from JigsawsGestureDataset import JigsawsGestureDataset
from JigsawsDatasetBase import JigsawsMetaDataset, ConcatDataset
from JigsawsConfig import main_config
from utils import torch_to_numpy
from position_encoding import PositionalEncoding
from TransformerModules import JigsawsFrameEncoder, JigsawsFrameDecoder
from utils import get_distance

import torch.optim as optim
from tqdm import tqdm
from vgg import Encoder, Decoder
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from losses import kl_criterion_normal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def get_encoded_frames_gestures(frame_encoder,frames,gestures,batch_size,seq_len):
  frames_batched = frames.view(batch_size*seq_len,frames.size(-3),frames.size(-2),frames.size(-1))

  # encode all frames
  encoded_frames, encoder_skips = frame_encoder(frames_batched)
  encoded_frames = encoded_frames.view(batch_size,seq_len,encoded_frames.size(-1))  

  encoder_skips = [skip.view(batch_size,seq_len,skip.size(-3),skip.size(-2),skip.size(-1)) for skip in encoder_skips]

  # prepare encoder and decoder inputs
  frames_gestures_past = torch.cat([encoded_frames[:,:past_count,:],gestures[:,:past_count,:]],dim=-1).to(device)

  frames_future_shifted = torch.roll(encoded_frames[:,past_count:,:], shifts=1, dims=1)
  frames_future_shifted[:,:1,:] = encoded_frames[:,past_count-1,:].unsqueeze(1)
  frames_gestures_future = torch.cat([frames_future_shifted,gestures[:,past_count:,:]],dim=-1).to(device)

  return frames_gestures_past, frames_gestures_future, encoder_skips



position_indices = main_config.kinematic_slave_position_indexes


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


batch_size = 4
num_epochs = 100
compressed_size = 128
subjects_num = 8
past_count = 10
future_count = 10
seq_len = past_count+future_count
num_gestures = 16
transformer_layers = 4
lr = 0.02
beta = 0.000000001

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((64,64)),
    # transforms.CenterCrop(64),
])

# Define dataset and dataloaders
df = pd.read_csv(os.path.join(main_config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject']=='C') & (df['Repetition']==1)].reset_index(drop=True)
df_valid = df[(df['Subject']=='C') & (df['Repetition']==1)].reset_index(drop=True)

dataset_train = ConcatDataset(JigsawsImageDataset(df_train,main_config,past_count+future_count,transform,sample_rate=6),
                        JigsawsGestureDataset(df_train,main_config,past_count+future_count,sample_rate=6),
                        JigsawsKinematicsDataset(df_train,main_config,past_count+future_count,sample_rate=6))
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_valid = ConcatDataset(JigsawsImageDataset(df_valid,main_config,past_count+future_count,transform,sample_rate=6),
                        JigsawsGestureDataset(df_valid,main_config,past_count+future_count,sample_rate=6),
                        JigsawsKinematicsDataset(df_valid,main_config,past_count+future_count,sample_rate=6))
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
frame_encoder = Encoder(compressed_size,3,None,'tanh').to(device)
frame_decoder = Decoder(compressed_size,3,None,'tanh').to(device)


encoder = JigsawsFrameEncoder(compressed_size+num_gestures,4,transformer_layers,compressed_size,0.1).to(device)
decoder = JigsawsFrameDecoder(compressed_size+num_gestures,4,transformer_layers,compressed_size,0.1,causal=True).to(device)

position_encoder = PositionalEncoding(compressed_size+num_gestures,0,past_count).to(device)

mse = nn.MSELoss()
mse_non_reduce = nn.MSELoss(reduce=False)

parameters = list(encoder.parameters()) + list(decoder.parameters()) +\
 list(frame_encoder.parameters())+\
 list(frame_decoder.parameters()) +\
 list(position_encoder.parameters())
optimizer = optim.Adam(parameters, lr=lr)

step_i=0

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    train_loss = torch.tensor([0.0,0.0,0.0])
    for batch in tqdm(dataloader_train):

        # Set the learning rate for this step
        step_i += 1
        # lr_temp = lr
        lr_temp = lr*min(step_i**-0.5,step_i*2000**-1.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_temp

        frames = batch[0].to(device)
        gestures = torch.nn.functional.one_hot(batch[1],num_gestures).to(device)
        positions = batch[2][:,:,position_indices].to(device)
        batch_size = frames.size(0)

        frame_encoder.batch_size = batch_size
        frame_decoder.batch_size = batch_size

        optimizer.zero_grad()

        frames_gestures_past, frames_gestures_future, encoder_skips = get_encoded_frames_gestures(frame_encoder,frames,gestures,batch_size,seq_len)
        # last_encoder_skips = [skip[:,past_count-1,:,:,:].view(batch_size*future_count,skip.size(-3),skip.size(-2),skip.size(-1)) for skip in encoder_skips]
        # last_encoder_skips = [torch.zeros(skip.shape).to(device) for skip in last_encoder_skips]
        last_encoder_skips = [skip[:,past_count-1,:,:,:] for skip in encoder_skips]

        # pass transformer encoder
        encoder_input = position_encoder(frames_gestures_past)
        encoder_output, encoder_output_mu, encoder_output_logvar = encoder(encoder_input)
        encoder_output = torch.cat([encoder_output, gestures[:,:past_count,:]],dim=-1)

        # pass transformer decoder iteratively
        decoder_input = torch.zeros(batch_size,future_count,compressed_size+num_gestures).to(device)
        decoder_output = None
        decoded_frames = []

        for i in range(future_count):
          if i==0:
            decoder_input[:,i,:compressed_size]=frames_gestures_past[:,-1,:compressed_size]
          else:
            decoder_input[:,i,:compressed_size]=frame_encoder(decoded_frames[i-1][:,0,:,:,:])[0]

          decoder_input[:,:i,compressed_size:] = gestures[:,past_count:(i+past_count),:]

          decoder_output,decoder_output_mu,decoder_output_logvar = decoder(position_encoder(decoder_input),encoder_output)
          decoder_output = decoder_output_mu
          decoded_frames.append(frame_decoder((decoder_output[:,i,:],last_encoder_skips)).unsqueeze(1))

        distance_weight_per_batch = get_distance(positions[:,:-1,:],positions[:,1:,:]).sum(dim=1)/seq_len
        cat_frames = torch.cat(decoded_frames,dim=1)
        mse_per_batch = mse_non_reduce(cat_frames, frames[:,past_count:,:,:,:]).sum(-1).sum(-1).sum(-1).sum(-1)
        loss_MSE = (distance_weight_per_batch*mse_per_batch).mean()
        loss_KDE = beta*kl_criterion_normal(encoder_output_mu,encoder_output_logvar).mean()

        loss_tot = loss_MSE + loss_KDE

        loss_tot.backward()
        optimizer.step()

        train_loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_KDE.item()])
        

    torch.cuda.empty_cache()
    valid_loss = 0.0
    for batch in tqdm(dataloader_valid):
        encoder.eval()
        decoder.eval()

        frames = batch[0].to(device)
        gestures = torch.nn.functional.one_hot(batch[1],num_gestures).to(device)
        positions = batch[2][:,:,position_indices].to(device)
        batch_size = frames.size(0)

        frames_gestures_past, frames_gestures_future, encoder_skips  = get_encoded_frames_gestures(frame_encoder,frames,gestures,batch_size,seq_len)
        # last_encoder_skips = [skip[:,past_count-1,:,:,:].unsqueeze(1).repeat(1,future_count,1,1,1).view(batch_size*future_count,skip.size(-3),skip.size(-2),skip.size(-1)) for skip in encoder_skips]
        # last_encoder_skips = [torch.zeros(skip.shape).to(device) for skip in last_encoder_skips]
        last_encoder_skips = [skip[:,past_count-1,:,:,:] for skip in encoder_skips]

        # pass transformer encoder
        encoder_input = position_encoder(frames_gestures_past)
        encoder_output, encoder_output_mu, encoder_output_logvar = encoder(encoder_input)
        encoder_output = torch.cat([encoder_output_mu, gestures[:,:past_count,:]],dim=-1)

        # pass transformer decoder iteratively
        decoder_input = torch.zeros(batch_size,future_count,compressed_size+num_gestures).to(device)
        decoder_output = None
        decoded_frames = []

        for i in range(future_count):
          if i==0:
            decoder_input[:,i,:compressed_size]=frames_gestures_past[:,-1,:compressed_size]
          else:
            decoder_input[:,i,:compressed_size]=frame_encoder(decoded_frames[i-1][:,0,:,:,:])[0]

          decoder_input[:,:i,compressed_size:] = gestures[:,past_count:(i+past_count),:]

          decoder_output,decoder_output_mu,decoder_output_logvar = decoder(position_encoder(decoder_input),encoder_output)
          decoder_output = decoder_output_mu
          decoded_frames.append(frame_decoder((decoder_output[:,i,:],last_encoder_skips)).unsqueeze(1))


        # decode output frames
        # decoded_frames = decoder_output.view(batch_size*future_count,compressed_size)
        # decoded_frames = frame_decoder((decoded_frames, last_encoder_skips))
        # decoded_frames = decoded_frames.view(batch_size,future_count,decoded_frames.size(-3),decoded_frames.size(-2),decoded_frames.size(-1))

        cat_frames = torch.cat(decoded_frames,dim=1)
        loss = mse(cat_frames, frames[:,past_count:,:,:,:])
        valid_loss += loss.item()
        break
        

    os.makedirs('/home/chen/MScProject/Code/experiments/Transformer_AutoEncoder/images/',exist_ok=True)

    fig = plt.figure(figsize=(10,4))
    frames_from_past_count = 3
    for i in range(future_count):
      plt.subplot(2,future_count+frames_from_past_count,frames_from_past_count+i+1)
      plt.imshow(torch_to_numpy(decoded_frames[i][0,0,:,:,:].detach()))
      plt.xticks([])
      plt.yticks([])
    for i in range(future_count+frames_from_past_count):
      plt.subplot(2,future_count+frames_from_past_count,i+1+future_count+frames_from_past_count)
      plt.imshow(torch_to_numpy(frames[0,past_count-frames_from_past_count+i,:,:,:].detach()))
      plt.title(batch[1][0,past_count-frames_from_past_count+i].item())
      plt.xticks([])
      plt.yticks([])


    plt.tight_layout()
    fig.savefig('/home/chen/MScProject/Code/experiments/Transformer_AutoEncoder/images/epoch_{}.png'.format(epoch))
    plt.close()

    train_loss /= len(dataloader_train)
    valid_loss /= len(dataloader_valid)


    print('Epoch {}: train loss {}'.format(epoch,train_loss.tolist()))
    print('Epoch {}: valid loss {}'.format(epoch,valid_loss))
    print('lr: {}'.format(lr_temp))


    
