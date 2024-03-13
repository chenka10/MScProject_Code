import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

models_dir = '/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder/models_position_based/' 

from JigsawsKinematicsDataset import JigsawsKinematicsDataset
from JigsawsImageDataset import JigsawsImageDataset
from JigsawsGestureDataset import JigsawsGestureDataset
from JigsawsDatasetBase import JigsawsMetaDataset, ConcatDataset
from JigsawsConfig import main_config
from utils import torch_to_numpy
import matplotlib.pyplot as plt

import torch.optim as optim
from tqdm import tqdm
from torchModules import BasicFc
from torchvision import models as tv_models
from vgg import Encoder, Decoder
from lstm import gaussian_lstm, lstm
from losses import kl_criterion_normal
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
import pandas as pd
from utils import get_distance


position_indices = main_config.kinematic_slave_position_indexes

import torch
import torch.nn as nn


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, input, target):
        return get_distance(input, target).mean()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 8
num_epochs = 100
img_compressed_size = 256
prior_size = 32
subjects_num = 8
past_count = 10
future_count = 10
seq_len = past_count+future_count
num_gestures = 16
added_vec_size = 24
transformer_layers = 4
lr = 0.0002
beta = 0.0001

start_epoch = 16

transform = transforms.Compose([
    transforms.ToTensor()
])

# Define dataset and dataloaders
df = pd.read_csv(os.path.join(main_config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject']!='C')].reset_index(drop=True)
df_valid = df[(df['Subject']=='C')].reset_index(drop=True)

dataset_train = ConcatDataset(JigsawsImageDataset(df_train,main_config,past_count+future_count,transform,sample_rate=6),
                        JigsawsGestureDataset(df_train,main_config,past_count+future_count,sample_rate=6),
                        JigsawsKinematicsDataset(df_train,main_config,past_count+future_count,sample_rate=6))
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_valid = ConcatDataset(JigsawsImageDataset(df_valid,main_config,past_count+future_count,transform,sample_rate=6),
                        JigsawsGestureDataset(df_valid,main_config,past_count+future_count,sample_rate=6),
                        JigsawsKinematicsDataset(df_valid,main_config,past_count+future_count,sample_rate=6))
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
frame_encoder = Encoder(img_compressed_size,3).to(device)
frame_decoder = Decoder(img_compressed_size,3).to(device)
prior_lstm = gaussian_lstm(img_compressed_size,prior_size,256,1,batch_size).to(device)
generation_lstm = lstm(img_compressed_size + prior_size + added_vec_size,img_compressed_size,256,2,batch_size).to(device)

if start_epoch > 0:
   frame_encoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_encoder.pth')))
   frame_decoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_decoder.pth')))
   prior_lstm.load_state_dict(torch.load(os.path.join(models_dir,'prior_lstm.pth')))
   generation_lstm.load_state_dict(torch.load(os.path.join(models_dir,'generation_lstm.pth')))

mse = nn.MSELoss(reduce=False)

models = [
    frame_encoder,
    frame_decoder,
    prior_lstm,
    generation_lstm    
]

parameters = sum([list(model.parameters()) for model in models],[])
optimizer = optim.Adam(parameters, lr=lr)

def expand_positions(positions):
   positions = positions.repeat(1,1,4)
   pos_multiplier = torch.tensor([1,1,1,1,1,1,10,10,10,10,10,10,100,100,100,100,100,100,1000,1000,1000,1000,1000,1000]).to(device)

   positions = positions*pos_multiplier

   return positions

# Training loop
for epoch in range(start_epoch, num_epochs):

    for model in models:
      model.train()

    train_loss = torch.tensor([0.0,0.0,0.0])
    for batch in tqdm(dataloader_train):

        frames = batch[0].to(device)
        gestures = torch.nn.functional.one_hot(batch[1],num_gestures).to(device)
        positions = batch[2][:,:,position_indices].to(device)
        positions_expanded = expand_positions(positions)
        batch_size = frames.size(0)

        # prepare lstms for batch
        prior_lstm.batch_size = batch_size
        prior_lstm.hidden = prior_lstm.init_hidden()

        generation_lstm.batch_size = batch_size
        generation_lstm.hidden = generation_lstm.init_hidden()

        optimizer.zero_grad()

        seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(seq_len)]
        generated_seq = []

        loss_MSE = torch.tensor(0.0).to(device)
        loss_KLD = torch.tensor(0.0).to(device)        

        distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:]).sum(dim=1)/seq_len
        batch_mover = distance_weight.argmax().item()

        for i in range(1,seq_len):
          frames_t = seq[i][0]

          if i <= past_count:
            frames_t_minus_one, skips = seq[i-1]
          else:
            frames_t_minus_one = seq[i-1][0]          

          z,mu,logvar = prior_lstm(frames_t_minus_one)
          frames_to_decode = generation_lstm(torch.cat([frames_t_minus_one,z,positions_expanded[:,i,:]],dim=-1))
          decoded_frames = frame_decoder([frames_to_decode,skips])
          generated_seq.append(decoded_frames)
          

          mse_per_batch = mse(decoded_frames, frames[:,i,:,:,:]).sum(-1).sum(-1).sum(-1)
          loss_MSE += (distance_weight*mse_per_batch).mean()
          loss_KLD += beta*kl_criterion_normal(mu,logvar) 
          

        loss_tot = loss_MSE + loss_KLD

        loss_tot.backward()
        optimizer.step()

        train_loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_KLD.item()])          
        

    valid_loss = torch.tensor([0.0,0.0,0.0])
    for batch in tqdm(dataloader_valid):        

        frames = batch[0].to(device)
        gestures = torch.nn.functional.one_hot(batch[1],num_gestures).to(device)
        positions = batch[2][:,:,position_indices].to(device)
        positions_expanded = expand_positions(positions)
        batch_size = frames.size(0)

        for model in models:
          model.eval()
          model.batch_size = batch_size

        # prepare lstms for batch
        prior_lstm.batch_size = batch_size
        prior_lstm.hidden = prior_lstm.init_hidden()

        generation_lstm.batch_size = batch_size
        generation_lstm.hidden = generation_lstm.init_hidden()

        seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(past_count)]
        generated_seq = []

        loss_MSE = torch.tensor(0.0).to(device)
        loss_KLD = torch.tensor(0.0).to(device)        

        distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:])
        batch_mover = distance_weight[:,past_count:].sum(dim=1).argmax().item()
        batch_least_mover = distance_weight[:,past_count:].sum(dim=1).argmin().item()
        distance_weight = distance_weight.sum(dim=1)/seq_len

        for i in range(1,seq_len):

          if i < past_count:
            frames_t = seq[i][0]
            frames_t_minus_one, skips = seq[i-1]
          elif i == past_count:
            frames_t = frame_encoder(decoded_frames)[0]
            frames_t_minus_one, skips = seq[i-1]
          else:
            frames_t = frame_encoder(decoded_frames)[0]
            frames_t_minus_one = frames_t
          
          z,mu,logvar = prior_lstm(frames_t_minus_one)
          frames_to_decode = generation_lstm(torch.cat([frames_t_minus_one,z,positions_expanded[:,i,:]],dim=-1))
          decoded_frames = frame_decoder([frames_to_decode,skips])

          generated_seq.append(decoded_frames)


          mse_per_batch = mse(decoded_frames, frames[:,i,:,:,:]).sum(-1).sum(-1).sum(-1)
          loss_MSE += (distance_weight*mse_per_batch).mean()
          loss_KLD += beta*kl_criterion_normal(mu,logvar)


        loss_tot = loss_MSE + loss_KLD
        valid_loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_KLD.item()])
        break

    os.makedirs('/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder/images_positions_based_2/',exist_ok=True)    
    os.makedirs(models_dir,exist_ok=True)

    torch.save(frame_encoder.state_dict(),os.path.join(models_dir,'frame_encoder.pth'))
    torch.save(frame_decoder.state_dict(),os.path.join(models_dir,'frame_decoder.pth'))
    torch.save(generation_lstm.state_dict(),os.path.join(models_dir,'generation_lstm.pth'))
    torch.save(prior_lstm.state_dict(),os.path.join(models_dir,'prior_lstm.pth'))

    fig = plt.figure(figsize=(10,4))
    frames_from_past_count = 3
    batch_to_show = batch_mover
    for i in range(future_count):
      plt.subplot(2,future_count+frames_from_past_count,frames_from_past_count+i+1)
      plt.imshow(torch_to_numpy(generated_seq[past_count-1+i][batch_to_show,:,:,:].detach()))
      plt.xticks([])
      plt.yticks([])
    for i in range(future_count+frames_from_past_count):
      plt.subplot(2,future_count+frames_from_past_count,i+1+future_count+frames_from_past_count)
      plt.imshow(torch_to_numpy(frames[batch_to_show,past_count-frames_from_past_count+i,:,:,:].detach()))
      plt.title(batch[1][batch_to_show,past_count-frames_from_past_count+i].item())
      plt.xticks([])
      plt.yticks([])

    plt.tight_layout()
    fig.savefig('/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder/images_positions_based_2/epoch_{}_mover.png'.format(epoch))
    plt.close()

    fig = plt.figure(figsize=(10,4))
    frames_from_past_count = 3
    batch_to_show = batch_least_mover
    for i in range(future_count):
      plt.subplot(2,future_count+frames_from_past_count,frames_from_past_count+i+1)
      plt.imshow(torch_to_numpy(generated_seq[past_count-1+i][batch_to_show,:,:,:].detach()))
      plt.xticks([])
      plt.yticks([])
    for i in range(future_count+frames_from_past_count):
      plt.subplot(2,future_count+frames_from_past_count,i+1+future_count+frames_from_past_count)
      plt.imshow(torch_to_numpy(frames[batch_to_show,past_count-frames_from_past_count+i,:,:,:].detach()))
      plt.title(batch[1][batch_to_show,past_count-frames_from_past_count+i].item())
      plt.xticks([])
      plt.yticks([])

    plt.tight_layout()
    fig.savefig('/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder/images_positions_based_2/epoch_{}_nonMover.png'.format(epoch))
    plt.close()

    train_loss /= len(dataloader_train)
    valid_loss /= len(dataloader_valid)


    print('Epoch {}: train loss {}'.format(epoch,train_loss.tolist()))
    print('Epoch {}: valid loss {}'.format(epoch,valid_loss.tolist()))    
