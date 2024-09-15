import sys
sys.path.append('/home/chen/MScProject/')
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import os
from JigsawsConfig import main_config as jigsaws_config

from visualizations import visualize_frame_diff
from models.blobReconstructor import BlobConfig, KinematicsToBlobs, PositionToBlobs, BlobsToFeatureMaps

import torch.optim as optim
from models.vgg import Encoder, MultiSkipsDecoder
from models.lstm import lstm
from models.blob_position_encoder import PositionEncoder
import os
from utils import get_distance
from datetime import datetime
import torch
import torch.nn as nn
from experiments.Blobs_LSTM.train_Blobs_LSTM_autoencoder import train
from experiments.Blobs_LSTM.validate_Blobs_LSTM_autoencoder import validate
from experiments.Blobs_LSTM.Blobs_LSTM_DataSetup import get_dataloaders
import random
import numpy as np
from utils import count_parameters


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, input, target):
        return get_distance(input, target).mean()

# 1. Set GPU to use
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('seed:', seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# 2. Set params
params = {
   'subject_to_leave':'D',
   'frame_size':64,
   'batch_size': 20,
   'num_epochs':20,
   'img_compressed_size': 256,
   'prior_size': 32,
   'subjects_num': 8,
   'past_count': 10,
   'future_count': 10,
   'num_gestures': 16,   
   'lr': 0.0005,
   'beta': 0.001,
   'gamma': 1000, 
   'conditioning':'position',
   'dataset':'JIGSAWS',
   'seed':seed
}
params['seq_len'] = params['past_count'] + params['future_count']

if params['dataset']=='ROSMA' and params['conditioning']!='position':
   raise ValueError('tried training on ROSMA dataset with conditioning that is not position.')

config = jigsaws_config

# 3. Setup data
dataloader_train, dataloader_valid, params = get_dataloaders(params,config)

# 4. Set if wandb should be used
use_wandb = True
start_epoch = 0
if use_wandb is True:
  import wandb
  wandb.login(key = '33514858884adc0292c3f8be3706845a1db35d3a')
  wandb.init(
     project = 'Robotic Surgery MSc',
     config = params,
     group = f'Next Frame Prediction - {params['conditioning']} Conditioned  - blobs, leave {params['subject_to_leave']}',
  )
  runid = wandb.run.id
else:
  runid = 3


now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

positions_to_blobs_dir = f'/home/chen/MScProject/Code/experiments/Blobs/2_blobs_seed_42_leave_{params['subject_to_leave']}_models'
models_dir = f'/home/chen/MScProject/Code/experiments/Blobs_LSTM/models_{params['conditioning']}/2_blobs_models_{timestamp}_leave_{params['subject_to_leave']}_{runid}/'
images_dir = f'/home/chen/MScProject/Code/experiments/Blobs_LSTM/images_{params['conditioning']}/2_blobs_images_{timestamp}_leave_{params['subject_to_leave']}_{runid}/'
os.makedirs(images_dir,exist_ok=True)    
os.makedirs(models_dir,exist_ok=True)

blob_feature_size = 16

# Initialize model, loss function, and optimizer 
frame_encoder = Encoder(params['img_compressed_size'],3).to(device)
frame_decoder = MultiSkipsDecoder(params['img_compressed_size'],blob_feature_size,3).to(device)
generation_lstm = lstm(params['img_compressed_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)

blob_config = [
    BlobConfig(0.25,0,4,[2,5],-torch.pi/7,'right'),
    BlobConfig(-0.25,0,4,[2,5],0,'left')
]
position_to_blobs = KinematicsToBlobs(blob_config)
position_to_blobs.load_state_dict(torch.load(os.path.join(positions_to_blobs_dir,'positions_to_blobs_14.pth')))
position_to_blobs.to(device)
blobs_to_maps = nn.ModuleList([BlobsToFeatureMaps(blob_feature_size,64),BlobsToFeatureMaps(blob_feature_size,64),                               
                               BlobsToFeatureMaps(blob_feature_size,32),BlobsToFeatureMaps(blob_feature_size,32),                                                              
                               BlobsToFeatureMaps(blob_feature_size,16),BlobsToFeatureMaps(blob_feature_size,16),
                               ]).to(device)

mse = nn.MSELoss(reduce=False)

models = [
    frame_encoder,
    frame_decoder,    
    generation_lstm,
    blobs_to_maps   
]

total_params = count_parameters(position_to_blobs)
for model in models:   
   total_params+=count_parameters(model)
print("total params: "+total_params)

parameters = sum([list(model.parameters()) for model in models],[])
optimizer = optim.Adam(parameters, lr=params['lr'])

for epoch in range(params['num_epochs']):

  # run train and validation loops
  train_loss, train_ssim_per_future_frame = train(models, position_to_blobs, dataloader_train, optimizer, params, config, device)

  with torch.no_grad():
    valid_loss, valid_ssim_per_future_frame, mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq = validate(models, position_to_blobs, dataloader_valid, params, config, device)    

  # save model weights  
  torch.save(frame_encoder.state_dict(),os.path.join(models_dir,f'frame_encoder_{epoch}.pth'))
  torch.save(frame_decoder.state_dict(),os.path.join(models_dir,f'frame_decoder_{epoch}.pth'))
  torch.save(generation_lstm.state_dict(),os.path.join(models_dir,f'generation_lstm_{epoch}.pth'))
  torch.save(blobs_to_maps.state_dict(),os.path.join(models_dir,f'blobs_to_maps_{epoch}.pth'))

  # save visualizations
  batch_seq_ind_to_save = [mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq]
  batch_seq_ind_names = ['mover','non-mover','best_mse','worst_mse']
  display_past_count = 3
  for i in range(len(batch_seq_ind_to_save)):
    batch, generated_seq, generated_grayscale_blob_maps, index = batch_seq_ind_to_save[i]
    frames = batch[0]
    gestures = batch[1]
    visualize_frame_diff(images_dir, batch_seq_ind_names[i], index, frames, generated_seq, generated_grayscale_blob_maps, display_past_count, params['past_count'], params['future_count'], epoch, gestures)  

  # print current results
  print('Epoch {}: train loss {}'.format(epoch,train_loss.tolist()))
  print('Epoch {}: valid loss {}'.format(epoch,valid_loss.tolist()))    
  print('Epoch {}: train ssim {}'.format(epoch,[round(val,4) for val in train_ssim_per_future_frame.round(decimals=4).tolist()]))
  print('Epoch {}: valid ssim {}'.format(epoch,[round(val,4) for val in valid_ssim_per_future_frame.round(decimals=4).tolist()]))    

  # log to wandb
  if use_wandb:
    data_to_log = {}
    for i in range(params['future_count']):
        data_to_log['train_SSIM_timestep_{}'.format(i)] = train_ssim_per_future_frame[i].item()
        data_to_log['valid_SSIM_timestep_{}'.format(i)] = valid_ssim_per_future_frame[i].item()
        
    data_to_log['train_MSE'] = train_loss[1].item()
    data_to_log['valid_MSE'] = valid_loss[1].item()
    # data_to_log['image'] = wandb.Image(image, caption=f"epoch {epoch}")    
    wandb.log(data_to_log)       