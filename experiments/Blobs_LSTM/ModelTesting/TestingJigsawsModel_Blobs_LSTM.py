import sys
sys.path.append('/home/chen/MScProject')
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')
sys.path.append('/home/chen/MScProject/Code/models')

from Code.models.blobReconstructor import BlobConfig, BlobsToFeatureMaps, PositionToBlobs, combine_blob_maps
from Code.models.lstm import lstm
from Code.models.vgg import Encoder, MultiSkipsDecoder


import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from Jigsaws.JigsawsGestureDataset import JigsawsGestureDataset
from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy
from modelUtils import load_models, iterate_on_images
from experiments.LSTM_AutoEncoder_Clean.DataSetup import unpack_batch
from tqdm import tqdm
import gc
import time
import torch.nn as nn

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

params = {
   'frame_size':64,
   'batch_size': 8,
   'num_epochs':100,
   'img_compressed_size': 256,
   'prior_size': 32,   
   'past_count': 10,
   'future_count': 30,
   'num_gestures': 16, 
   'conditioning':'position', #'gesture'   
   'dataset':'JIGSAWS'
}
params['seq_len'] = params['past_count'] + params['future_count']

taskId = 2
subject = 'C'
repetition = 3

past_count = params['past_count']
future_count = params['future_count']

transform = transforms.Compose([
    transforms.ToTensor()
])
df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_test = df[(df['Subject']=='C') & (df['Repetition']==repetition)].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,2,transform,sample_rate=jigsaws_sample_rate),
                        JigsawsGestureDataset(df_test,config,2,sample_rate=jigsaws_sample_rate),
                        JigsawsKinematicsDataset(df_test,config,2,sample_rate=jigsaws_sample_rate))


# load models
models_dir = '/home/chen/MScProject/Code/experiments/Blobs_LSTM/models_position/models_20240530_214809_1'

blob_feature_size = 16

frame_encoder = Encoder(params['img_compressed_size'],3)
frame_encoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_encoder.pth')))
frame_encoder.to(device)
frame_encoder.eval()
frame_encoder.batch_size = 1

frame_decoder = MultiSkipsDecoder(params['img_compressed_size'],blob_feature_size,3).to(device)
frame_decoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_decoder.pth')))
frame_decoder.to(device)
frame_decoder.eval()
frame_decoder.batch_size = 1

generation_lstm = lstm(params['img_compressed_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)
generation_lstm.load_state_dict(torch.load(os.path.join(models_dir,'generation_lstm.pth')))
generation_lstm.to(device)
generation_lstm.eval()
generation_lstm.batch_size = 1


blob_config = [
    BlobConfig(0.25,0,4,[2,4],'right'),
    BlobConfig(-0.25,0,4,[2,4],'left')
]
position_to_blobs = PositionToBlobs(blob_config)
positions_to_blobs_dir = '/home/chen/MScProject/Code/experiments/Blobs/seed_3_42_models'
position_to_blobs.load_state_dict(torch.load(os.path.join(positions_to_blobs_dir,'positions_to_blobs.pth')))
position_to_blobs.to(device)
position_to_blobs.eval()

blobs_to_maps = nn.ModuleList([BlobsToFeatureMaps(blob_feature_size,64),BlobsToFeatureMaps(blob_feature_size,64),
                               BlobsToFeatureMaps(blob_feature_size,32),BlobsToFeatureMaps(blob_feature_size,32),
                               BlobsToFeatureMaps(blob_feature_size,16),BlobsToFeatureMaps(blob_feature_size,16)]).to(device)
blobs_to_maps.load_state_dict(torch.load(os.path.join(models_dir,'blobs_to_maps.pth')))
blobs_to_maps.to(device)
blobs_to_maps.eval()


frames_dir = f'/home/chen/MScProject/Code/experiments/Blobs_LSTM/ModelTesting/V1_3_{taskId}_{subject}_{repetition}_{params['conditioning']}'
os.makedirs(frames_dir, exist_ok=True)

generation_lstm.hidden = generation_lstm.init_hidden()

for t in tqdm(range(len(dataset_test))):

   batch = [b.unsqueeze(0) for b in dataset_test[t]]   
   
   frames, gestures, gestures_onehot, positions, rotations, kinematics, _ = unpack_batch(params, config, batch, device)           

   blob_datas = position_to_blobs(positions[:,1,:])
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
   if t <= params['past_count']:          
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

   fig, axes = plt.subplots(1,2)
   
   axes[0].imshow(torch_to_numpy(decoded_frames[0,:,:,:].detach()))
   axes[1].imshow(torch_to_numpy(frames[0,1,:,:,:].detach()))
   
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

   