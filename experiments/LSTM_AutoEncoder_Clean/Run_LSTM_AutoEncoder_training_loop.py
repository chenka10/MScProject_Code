import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import os
from JigsawsConfig import main_config as jigsaws_config
from ROSMA.RosmaConfig import config as rosma_config

from visualizations import visualize_frame_diff


import torch.optim as optim
from models.vgg import Encoder, Decoder
from models.vgg128 import Encoder128, Decoder128
from models.lstm import gaussian_lstm, lstm
import os
from utils import get_distance
from datetime import datetime
import torch
import torch.nn as nn
from train_lstm_autoencoder import train
from validate_lstm_autoencoder import validate

from DataSetup import get_dataloaders


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, input, target):
        return get_distance(input, target).mean()

# 1. Set GPU to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 2. Set params
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
   'gamma': 1000, 
   'conditioning':'position', #'gesture'   
   'dataset':'ROSMA'
}
params['seq_len'] = params['past_count'] + params['future_count']
if params['conditioning'] == 'position':
   params['added_vec_size'] = 32
elif params['conditioning'] == 'gesture':
   params['added_vec_size'] = 16
else:
   raise ValueError()

if params['dataset']=='ROSMA' and params['conditioning']!='position':
   raise ValueError('tried training on ROSMA dataset with conditioning that is not position.')

if params['dataset'] == 'JIGSAWS': config = jigsaws_config
else: config = rosma_config

# 3. Set if wandb should be used
use_wandb = True
start_epoch = 0
if use_wandb is True:
  import wandb
  wandb.login(key = '33514858884adc0292c3f8be3706845a1db35d3a')
  wandb.init(
     project = 'Robotic Surgery MSc',
     config = params,
     group = f'Next Frame Prediction - {params['conditioning']} Conditioned (with rotation) (Stochastic inference)',     
  )
  runid = wandb.run.id
else:
  runid = 3

# 4. Setup data
dataloader_train, dataloader_valid = get_dataloaders(params,config)

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

models_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/models_{params['conditioning']}/models_{timestamp}_{runid}/'
images_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/images_{params['conditioning']}/images_{timestamp}_{runid}/'
os.makedirs(images_dir,exist_ok=True)    
os.makedirs(models_dir,exist_ok=True)


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
  train_loss, train_ssim_per_future_frame = train(models, dataloader_train, optimizer, params, config, device)

  with torch.no_grad():
    valid_loss, valid_ssim_per_future_frame, mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq = validate(models, dataloader_valid, params, config, device)    

  # save model weights  
  torch.save(frame_encoder.state_dict(),os.path.join(models_dir,'frame_encoder.pth'))
  torch.save(frame_decoder.state_dict(),os.path.join(models_dir,'frame_decoder.pth'))
  torch.save(generation_lstm.state_dict(),os.path.join(models_dir,'generation_lstm.pth'))
  torch.save(prior_lstm.state_dict(),os.path.join(models_dir,'prior_lstm.pth'))

  # save visualizations
  batch_seq_ind_to_save = [mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq]
  batch_seq_ind_names = ['mover','non-mover','best_mse','worst_mse']
  display_past_count = 3
  for i in range(len(batch_seq_ind_to_save)):
    batch, generated_seq, index = batch_seq_ind_to_save[i]
    frames = batch[0]
    gestures = batch[1]
    visualize_frame_diff(images_dir, batch_seq_ind_names[i], index, frames, generated_seq, display_past_count, params['past_count'], params['future_count'], epoch, gestures)  

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