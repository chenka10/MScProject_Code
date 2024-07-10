import sys
sys.path.append('/home/chen/MScProject')
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')
sys.path.append('/home/chen/MScProject/Code/models')

from Code.experiments.rarp50.Blobs_LSTM.rarp50_Blobs_LSTM_DataSetup import unpack_batch_rarp50
from Code.models.blobReconstructor import BlobConfig, BlobsToFeatureMaps, KinematicsToBlobs, PositionToBlobs, combine_blob_maps
from Code.models.lstm import lstm
from Code.models.vgg import Encoder, MultiSkipsDecoder
from Code.models.vgg128 import Encoder128, MultiSkipsDecoder128
from Code.rarp50.rarp50Config import config
from Code.rarp50.rarp50ImageDataset import rarp50ImageDataset
from Code.rarp50.rarp50KinematicsDataset import rarp50KinematicsDataset


import pandas as pd
import os
from DataUtils import ConcatDataset
from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy
from experiments.Blobs_LSTM.Blobs_LSTM_DataSetup import unpack_batch
from tqdm import tqdm
import gc
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

params = {
   'frame_size':128,
   'batch_size': 1,
   'num_epochs':100,
   'img_compressed_size': 256,
   'prior_size': 32,   
   'past_count': 100,
   'future_count': 30,
   'num_gestures': 16, 
   'conditioning':'position', #'gesture'   
   'dataset':'JIGSAWS'
}
params['seq_len'] = params['past_count'] + params['future_count']

video = 37

DIGITS_IN_SEGMENTATION_FILE_NAME = 5
FRAME_INCREMENT = 12

config.rarp50_videoFramesDir = os.path.join(config.project_baseDir,f'data/rarp50_{params['frame_size']}')    

df = pd.read_csv(os.path.join('/home/chen/MScProject/rarp50_filtered_data_detailed.csv'))
df = df[df['videoName'].isin([f'video_{video}'])].reset_index(drop=True)

SEQUENCE_LENGTH = 2

dataset = ConcatDataset(rarp50ImageDataset(df,config,SEQUENCE_LENGTH,DIGITS_IN_SEGMENTATION_FILE_NAME,FRAME_INCREMENT),rarp50KinematicsDataset(df,config,SEQUENCE_LENGTH,FRAME_INCREMENT))
dataloader = DataLoader(dataset,params['batch_size'],True,drop_last=True)

# load models
positions_to_blobs_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs/2_blobs_frameSize_128_seed_42_models'
models_dir = '/home/chen/MScProject/Code/experiments/rarp50/Blobs_LSTM/models_position/2_blobs_framesize_128_leave_37_models_20240629_174022_i1mrwoir'
epoch = 21

blob_feature_size = 16

if params['frame_size']==128:
   frame_encoder = Encoder128(params['img_compressed_size'],3)
   frame_encoder.load_state_dict(torch.load(os.path.join(models_dir,f'frame_encoder_{epoch}.pth')))
   frame_encoder.to(device)
   frame_encoder.eval()
   frame_encoder.batch_size = 1

   frame_decoder = MultiSkipsDecoder128(params['img_compressed_size'],blob_feature_size,3).to(device)
   frame_decoder.load_state_dict(torch.load(os.path.join(models_dir,f'frame_decoder_{epoch}.pth')))
   frame_decoder.to(device)
   frame_decoder.eval()
   frame_decoder.batch_size = 1
else:
   raise ValueError('params["frame_size"] must be 128')

generation_lstm = lstm(params['img_compressed_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)
generation_lstm.load_state_dict(torch.load(os.path.join(models_dir,f'generation_lstm_{epoch}.pth')))
generation_lstm.to(device)
generation_lstm.eval()
generation_lstm.batch_size = 1


blob_config = [
    BlobConfig(0.25,0,0,[1,10],0,'right'),
    BlobConfig(-0.25,0,0,[1,10],0,'left')
]
POSITION_TO_BLOBS_MODEL_EPOCH = 99
position_to_blobs = KinematicsToBlobs(blob_config,True,True)
position_to_blobs.load_state_dict(torch.load(os.path.join(positions_to_blobs_dir,f'positions_to_blobs_{POSITION_TO_BLOBS_MODEL_EPOCH}.pth')))
position_to_blobs.to(device)
img_size = params['frame_size']
blobs_to_maps = nn.ModuleList([BlobsToFeatureMaps(blob_feature_size,img_size),BlobsToFeatureMaps(blob_feature_size,img_size),                               
                               BlobsToFeatureMaps(blob_feature_size,img_size/2),BlobsToFeatureMaps(blob_feature_size,img_size/2),        
                               BlobsToFeatureMaps(blob_feature_size,img_size/4),BlobsToFeatureMaps(blob_feature_size,img_size/4),
                               ]).to(device)
blobs_to_maps.eval()


frames_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs_LSTM/ModelTesting/2_blobs_framesize_128_leave_37_models_20240629_174022_i1mrwoir'
os.makedirs(frames_dir, exist_ok=True)

generation_lstm.hidden = generation_lstm.init_hidden()

START_FRAME = 30
for t in tqdm(range(START_FRAME, len(dataset))):     
   
   batch = dataset[t]

   frames, kinematics,ecm_kinematics, positions, batch_size = unpack_batch_rarp50(batch, device)

   # adding artificial batch dimension
   frames = frames.unsqueeze(0)
   kinematics = kinematics.unsqueeze(0)
   ecm_kinematics = ecm_kinematics.unsqueeze(0)
   positions = positions.unsqueeze(0)   

   blob_datas = position_to_blobs(kinematics[:,1,:],ecm_kinematics = ecm_kinematics[:,1,:])
   feature_maps = []
   grayscale_maps = []

   for i in range(len(blobs_to_maps)):
      f,g = blobs_to_maps[i](blob_datas[i%2])
      feature_maps.append(f)
      grayscale_maps.append(g)
   
   combined_blobs_feature_maps = []
   for i in range(len(blobs_to_maps)//2):
      combined_blobs_feature_maps.append(combine_blob_maps(torch.zeros_like(feature_maps[i*2]),
                                                      [feature_maps[i*2],feature_maps[i*2+1]],
                                                      [grayscale_maps[i*2],grayscale_maps[i*2+1]]))

   # keep loading past frames (for conditioning), once conditioning is over, load previously encoded frames
   if (t-START_FRAME) <= params['past_count']:          
      frames_t_minus_one, skips = frame_encoder(frames[:,0,:,:,:])
      skips[0] = torch.concat([skips[0],combined_blobs_feature_maps[0]],1)
      skips[1] = torch.concat([skips[1],combined_blobs_feature_maps[1]],1)
      skips[2] = torch.concat([skips[2],combined_blobs_feature_maps[2]],1)
   else:          
      frames_t_minus_one = frame_encoder(decoded_frames.to(device))[0]
      skips[0][:,-combined_blobs_feature_maps[0].size(1):,:,:] = combined_blobs_feature_maps[0]
      skips[1][:,-combined_blobs_feature_maps[1].size(1):,:,:] = combined_blobs_feature_maps[1]
      skips[2][:,-combined_blobs_feature_maps[2].size(1):,:,:] = combined_blobs_feature_maps[2]          
   

   # predict next frame latent, decode next frame, store next frame
   frames_to_decode = generation_lstm(frames_t_minus_one).float()
   decoded_frames = frame_decoder([frames_to_decode,skips]).cpu()

   fig, axes = plt.subplots(1,3)
   
   axes[0].imshow(torch_to_numpy(decoded_frames[0,:,:,:].detach()))
   axes[1].imshow(torch_to_numpy(frames[0,1,:,:,:].detach()))
   axes[2].imshow(torch_to_numpy(grayscale_maps[0][0].detach()) + torch_to_numpy(grayscale_maps[1][0].detach()))
   
   plt.tight_layout()
   plt.savefig(os.path.join(frames_dir,f'test_{t}.png'))
   plt.close()   

   frames = None
   gestures = None
   gestures_onehot = None
   positions = None
   rotations = None
   kinematics = None
   conditioning_vec = None
   batch = None
   frames_to_decode = None
   z = None
   mu = None
   logvar = None
   frames_t_minus_one = None   
   del frames, gestures, gestures_onehot, positions, rotations, kinematics, conditioning_vec, batch, frames_to_decode,z,mu,logvar, frames_t_minus_one

   gc.collect()   
   torch.cuda.empty_cache()

   