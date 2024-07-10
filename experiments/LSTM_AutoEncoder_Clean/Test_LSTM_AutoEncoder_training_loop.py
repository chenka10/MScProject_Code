import random
import sys

import numpy as np
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

def find_directory(base_dir, search_string):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if search_string in dir_name:
                return os.path.join(root, dir_name)
    return None

# 1. Set GPU to use
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('seed:', seed)

runs_by_subject = { 
   'B':'lob314nf',
   'C':'kh4d1ek2',
   'D':'cg7poged',
   'E':'4ljrmbff',
   'F':'iss6qjlg',
   'G':'4qu8d6zv',
   'H':'ihjq13i1',
   'I':'l7yeh4hd'
}
subject_to_leave = 'D'

# 2. Set params
params = {
   'frame_size':64,
   'batch_size': 8,
   'num_epochs':100,
   'img_compressed_size': 256,
   'prior_size': 32,
   'subjects_num': 8,
   'past_count': 10,
   'future_count': 20,
   'num_gestures': 16,   
   'lr': 0.0005,
   'beta': 0.001,
   'gamma': 1000, 
   'conditioning':'position', #'gesture'   
   'dataset':'JIGSAWS',
   'leave_subject':subject_to_leave,
   'orig_runid': runs_by_subject[subject_to_leave],   
}
params['seq_len'] = params['past_count'] + params['future_count']
if params['conditioning'] == 'position':
   params['added_vec_size'] = 42 # 24 - expanded position, 18 - rotation matrices
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

models_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/test_models_{params['conditioning']}/models_{timestamp}_leave_{params['leave_subject']}_{runid}_origis_{params['orig_runid']}/'
images_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/test_images_{params['conditioning']}/images_{timestamp}_leave_{params['leave_subject']}_{runid}_origis_{params['orig_runid']}/'
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


orig_models_dir = find_directory('/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/models_position',params['orig_runid'])

frame_encoder.load_state_dict(torch.load(os.path.join(orig_models_dir,f'frame_encoder.pth')))
frame_decoder.load_state_dict(torch.load(os.path.join(orig_models_dir,f'frame_decoder.pth')))
prior_lstm.load_state_dict(torch.load(os.path.join(orig_models_dir,f'prior_lstm.pth')))
generation_lstm.load_state_dict(torch.load(os.path.join(orig_models_dir,f'generation_lstm.pth')))

models = [
    frame_encoder,
    frame_decoder,
    prior_lstm,
    generation_lstm    
]

# parameters = sum([list(model.parameters()) for model in models],[])
# optimizer = optim.Adam(parameters, lr=params['lr'])

with torch.no_grad():
  valid_loss, valid_ssim_per_future_frame, mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq = validate(models, dataloader_valid, params, config, device)    

# save visualizations
# batch_seq_ind_to_save = [mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq, worst_batch_seq]
# batch_seq_ind_names = ['mover','non-mover','best_mse','worst_mse']
# display_past_count = 3
# for i in range(len(batch_seq_ind_to_save)):
#   batch, generated_seq, index = batch_seq_ind_to_save[i]
#   frames = batch[0]
#   gestures = batch[1]
#   visualize_frame_diff(images_dir, batch_seq_ind_names[i], index, frames, generated_seq, None, display_past_count, params['past_count'], params['future_count'], epoch, gestures)  

# log to wandb
if use_wandb:
  data_to_log = {}
  for i in range(params['future_count']):      
      data_to_log['valid_SSIM_timestep_{}'.format(i)] = valid_ssim_per_future_frame[i].item()      
  
  data_to_log['valid_MSE'] = valid_loss[1].item()
  # data_to_log['image'] = wandb.Image(image, caption=f"epoch {epoch}")    
  wandb.log(data_to_log)       