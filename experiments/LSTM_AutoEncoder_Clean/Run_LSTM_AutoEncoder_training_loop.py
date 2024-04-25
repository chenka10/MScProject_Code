import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import os

import wandb
wandb.login(key = '33514858884adc0292c3f8be3706845a1db35d3a')

from JigsawsKinematicsDataset import JigsawsKinematicsDataset
from JigsawsImageDataset import JigsawsImageDataset
from JigsawsGestureDataset import JigsawsGestureDataset
from JigsawsDatasetBase import ConcatDataset
from JigsawsConfig import main_config
from utils import torch_to_numpy
import matplotlib.pyplot as plt


import torch.optim as optim
from models.vgg import Encoder, Decoder
from models.vgg128 import Encoder128, Decoder128
from models.lstm import gaussian_lstm, lstm
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
import pandas as pd
from utils import get_distance
import io
from PIL import Image
from datetime import datetime

position_indices = main_config.kinematic_slave_position_indexes

import torch
import torch.nn as nn
from train_lstm_autoencoder import train
from validate_lstm_autoencoder import validate


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, input, target):
        return get_distance(input, target).mean()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv(os.path.join(main_config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject']!='C')].reset_index(drop=True)
df_valid = df[(df['Subject']=='C')].reset_index(drop=True)

params = {
   'frame_size':128,
   'batch_size': 8,
   'num_epochs':100,
   'img_compressed_size': 256,
   'prior_size': 32,
   'subjects_num': 8,
   'past_count': 10,
   'future_count': 10,
   'num_gestures': 16,   
   'lr': 0.0005,
   'beta': 0.001,    
   'conditioning':'position' #'gesture'
}
params['seq_len'] = params['past_count'] + params['future_count']
if params['conditioning'] == 'position':
   params['added_vec_size'] = 32
elif params['conditioning'] == 'gesture':
   params['added_vec_size'] = 16
else:
   raise ValueError()

params['train_subjects'] = df_train['Subject'].unique()
params['train_repetitions'] = df_train['Repetition'].unique()
params['valid_subjects'] = df_valid['Subject'].unique()
params['valid_repetitions'] = df_valid['Repetition'].unique()

if params['frame_size'] == 128:
   main_config.extracted_frames_dir = '/home/chen/MScProject/data/jigsaws_extracted_frames_128/'

start_epoch = 0
use_wandb = True
if use_wandb:
  wandb.init(
     project = 'Robotic Surgery MSc',
     config = params,
     group = f'Next Frame Prediction - {params['conditioning']} Conditioned (with rotation) (Stochastic inference)',     
  )
  runid = wandb.run.id
else:
  runid = 3

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

models_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/models_{params['conditioning']}/models_{timestamp}_{runid}/'
images_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/images_{params['conditioning']}/images_{timestamp}_{runid}/'
os.makedirs(images_dir,exist_ok=True)    
os.makedirs(models_dir,exist_ok=True)

# Define dataset and dataloaders
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset_train = ConcatDataset(JigsawsImageDataset(df_train,main_config,params['past_count']+params['future_count'],transform,sample_rate=6),
                        JigsawsGestureDataset(df_train,main_config,params['past_count']+params['future_count'],sample_rate=6),
                        JigsawsKinematicsDataset(df_train,main_config,params['past_count']+params['future_count'],sample_rate=6))
dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)

dataset_valid = ConcatDataset(JigsawsImageDataset(df_valid,main_config,params['past_count']+params['future_count'],transform,sample_rate=6),
                        JigsawsGestureDataset(df_valid,main_config,params['past_count']+params['future_count'],sample_rate=6),
                        JigsawsKinematicsDataset(df_valid,main_config,params['past_count']+params['future_count'],sample_rate=6))
dataloader_valid = DataLoader(dataset_valid, batch_size=params['batch_size'], shuffle=True)

# Initialize model, loss function, and optimizer
if params['frame_size'] == 128:
  frame_encoder = Encoder128(params['img_compressed_size'],3).to(device)
  frame_decoder = Decoder128(params['img_compressed_size'],3).to(device)
else:   
  frame_encoder = Encoder(params['img_compressed_size'],3).to(device)
  frame_decoder = Decoder(params['img_compressed_size'],3).to(device)

prior_lstm = gaussian_lstm(params['img_compressed_size'],params['prior_size'],256,1,params['batch_size'],device).to(device)
generation_lstm = lstm(params['img_compressed_size'] + params['prior_size'] + params['added_vec_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)

mse = nn.MSELoss(reduce=False)

models = [
    frame_encoder,
    frame_decoder,
    prior_lstm,
    generation_lstm    
]

parameters = sum([list(model.parameters()) for model in models],[])
optimizer = optim.Adam(parameters, lr=params['lr'])

for epoch in range(start_epoch, params['num_epochs']):

  # run train and validation loops
  train_loss, train_ssim_per_future_frame = train(models, dataloader_train, optimizer, params, device)

  with torch.no_grad():
    valid_loss, valid_ssim_per_future_frame, batch_mover, batch_least_mover, generated_seq, frames, batch = validate(models, dataloader_valid, params, device)
    
  torch.save(frame_encoder.state_dict(),os.path.join(models_dir,'frame_encoder.pth'))
  torch.save(frame_decoder.state_dict(),os.path.join(models_dir,'frame_decoder.pth'))
  torch.save(generation_lstm.state_dict(),os.path.join(models_dir,'generation_lstm.pth'))
  torch.save(prior_lstm.state_dict(),os.path.join(models_dir,'prior_lstm.pth'))

  fig = plt.figure(figsize=(10,4))
  frames_from_past_count = 3
  batch_to_show = batch_mover
  for i in range(params['future_count']):
    plt.subplot(2,params['future_count']+frames_from_past_count,frames_from_past_count+i+1)
    plt.imshow(torch_to_numpy(generated_seq[params['past_count']-1+i][batch_to_show,:,:,:].detach()))
    plt.xticks([])
    plt.yticks([])
  for i in range(params['future_count']+frames_from_past_count):
    plt.subplot(2,params['future_count']+frames_from_past_count,i+1+params['future_count']+frames_from_past_count)
    plt.imshow(torch_to_numpy(frames[batch_to_show,params['past_count']-frames_from_past_count+i,:,:,:].detach()))
    plt.title(batch[1][batch_to_show,params['past_count']-frames_from_past_count+i].item())
    plt.xticks([])
    plt.yticks([])

  plt.tight_layout()
  fig.savefig(os.path.join(images_dir,'epoch_{}_mover.png').format(epoch))

  # Save the figure to a BytesIO object
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png')
  buffer.seek(0)

  # Open the BytesIO object as a PIL image
  image = Image.open(buffer)
  plt.close()

  fig = plt.figure(figsize=(10,4))
  frames_from_past_count = 3
  batch_to_show = batch_least_mover
  for i in range(params['future_count']):
    plt.subplot(2,params['future_count']+frames_from_past_count,frames_from_past_count+i+1)
    plt.imshow(torch_to_numpy(generated_seq[params['past_count']-1+i][batch_to_show,:,:,:].detach()))
    plt.xticks([])
    plt.yticks([])
  for i in range(params['future_count']+frames_from_past_count):
    plt.subplot(2,params['future_count']+frames_from_past_count,i+1+params['future_count']+frames_from_past_count)
    plt.imshow(torch_to_numpy(frames[batch_to_show,params['past_count']-frames_from_past_count+i,:,:,:].detach()))
    plt.title(batch[1][batch_to_show,params['past_count']-frames_from_past_count+i].item())
    plt.xticks([])
    plt.yticks([])

  plt.tight_layout()
  fig.savefig(os.path.join(images_dir,'epoch_{}_nonMover.png').format(epoch))
  plt.close() 

  print('Epoch {}: train loss {}'.format(epoch,train_loss.tolist()))
  print('Epoch {}: valid loss {}'.format(epoch,valid_loss.tolist()))    

  print('Epoch {}: train ssim {}'.format(epoch,[round(val,4) for val in train_ssim_per_future_frame.round(decimals=4).tolist()]))
  print('Epoch {}: valid ssim {}'.format(epoch,[round(val,4) for val in valid_ssim_per_future_frame.round(decimals=4).tolist()]))    

  if use_wandb:
    data_to_log = {}
    for i in range(params['future_count']):
        data_to_log['train_SSIM_timestep_{}'.format(i)] = train_ssim_per_future_frame[i].item()
        data_to_log['valid_SSIM_timestep_{}'.format(i)] = valid_ssim_per_future_frame[i].item()
        
    data_to_log['train_MSE'] = train_loss[1].item()
    data_to_log['valid_MSE'] = valid_loss[1].item()
    data_to_log['image'] = wandb.Image(image, caption=f"epoch {epoch}")    
    wandb.log(data_to_log)       


