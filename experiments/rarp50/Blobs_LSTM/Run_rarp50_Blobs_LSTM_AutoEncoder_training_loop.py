import sys
sys.path.append('/home/chen/MScProject/')
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import os


from visualizations import visualize_frame_diff
from models.blobReconstructor import BlobConfig, KinematicsToBlobs, PositionToBlobs, BlobsToFeatureMaps

from Code.experiments.rarp50.Blobs_LSTM.train_rarp50_Blobs_LSTM_autoencoder import train
from Code.experiments.rarp50.Blobs_LSTM.validate_rarp50_Blobs_LSTM_autoencoder import validate
from Code.DataUtils import ConcatDataset
from Code.models.vgg128 import Encoder128, MultiSkipsDecoder128
from Code.rarp50.rarp50Config import config
from Code.rarp50.rarp50ImageDataset import rarp50ImageDataset
from Code.rarp50.rarp50KinematicsDataset import rarp50KinematicsDataset

import torch.optim as optim
from torch.utils.data import DataLoader
from models.vgg import Encoder, MultiSkipsDecoder
from models.lstm import lstm
from models.blob_position_encoder import PositionEncoder
import os
from utils import get_distance
from datetime import datetime
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd



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
   'frame_size':64,
   'batch_size': 22,
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
   'dataset':'rarp50',
   'seed':seed
}
params['seq_len'] = params['past_count'] + params['future_count']

if params['dataset']=='ROSMA' and params['conditioning']!='position':
   raise ValueError('tried training on ROSMA dataset with conditioning that is not position.')


# 3. Setup data
df = pd.read_csv(os.path.join('/home/chen/MScProject/rarp50_filtered_data_detailed.csv'))
df_train = df[~df['videoName'].isin(['video_37'])].reset_index(drop=True)
df_test = df[df['videoName'].isin(['video_37'])].reset_index(drop=True)

DIGITS_IN_SEGMENTATION_FILE_NAME = 5
FRAME_INCREMENT = 12

config.rarp50_videoFramesDir = os.path.join(config.project_baseDir,f'data/rarp50_{params['frame_size']}')    

dataset_train = ConcatDataset(rarp50ImageDataset(df_train,config,params['seq_len'],DIGITS_IN_SEGMENTATION_FILE_NAME,FRAME_INCREMENT),rarp50KinematicsDataset(df_train,config,params['seq_len'],FRAME_INCREMENT))
dataloader_train = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

dataset_test = ConcatDataset(rarp50ImageDataset(df_test,config,params['seq_len'],DIGITS_IN_SEGMENTATION_FILE_NAME,FRAME_INCREMENT),rarp50KinematicsDataset(df_test,config,params['seq_len'],FRAME_INCREMENT))
dataloader_valid = DataLoader(dataset_test,params['batch_size'],True,drop_last=True)


# 4. Set if wandb should be used
use_wandb = False
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

positions_to_blobs_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs/2_blobs_extendedBlobWindow_frameSize_128_seed_42_models'
models_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs_LSTM/models_{params['conditioning']}/2_blobs_models_{timestamp}_{runid}/'
images_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs_LSTM/images_{params['conditioning']}/2_blobs_images_{timestamp}_{runid}/'
os.makedirs(images_dir,exist_ok=True)    
os.makedirs(models_dir,exist_ok=True)

blob_feature_size = 16

# Initialize model, loss function, and optimizer 
if params['frame_size'] == 64:
  frame_encoder = Encoder(params['img_compressed_size'],3).to(device)
  frame_decoder = MultiSkipsDecoder(params['img_compressed_size'],blob_feature_size,nc = 3).to(device)
elif params['frame_size']==128:
  frame_encoder = Encoder128(params['img_compressed_size'],3).to(device)
  frame_decoder = MultiSkipsDecoder128(params['img_compressed_size'],blob_feature_size,nc = 3).to(device)
else:
   raise ValueError('params["frame_size"] must be either 64 or 128')

generation_lstm = lstm(params['img_compressed_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)

# start_x: float,
#     start_y: int,
#     start_s: int,
#     a_range: list[int],
#     start_theta: int,
#     side: str
blob_config = [
    BlobConfig(0.25,0,6,[1,10],0,'right'),
    BlobConfig(-0.25,0,6,[1,10],0,'left')
]


POSITION_TO_BLOBS_MODEL_EPOCH = 14
position_to_blobs = KinematicsToBlobs(blob_config,True)
position_to_blobs.load_state_dict(torch.load(os.path.join(positions_to_blobs_dir,f'positions_to_blobs_{POSITION_TO_BLOBS_MODEL_EPOCH}.pth')))
position_to_blobs.to(device)
img_size = params['frame_size']
blobs_to_maps = nn.ModuleList([BlobsToFeatureMaps(blob_feature_size,img_size),BlobsToFeatureMaps(blob_feature_size,img_size),                               
                               BlobsToFeatureMaps(blob_feature_size,img_size/2),BlobsToFeatureMaps(blob_feature_size,img_size/2),        
                               BlobsToFeatureMaps(blob_feature_size,img_size/4),BlobsToFeatureMaps(blob_feature_size,img_size/4),
                               ]).to(device)

mse = nn.MSELoss(reduce=False)

models = [
    frame_encoder,
    frame_decoder,    
    generation_lstm,
    blobs_to_maps   
]

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