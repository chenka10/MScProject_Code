import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')

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
if params['conditioning'] == 'position':
   params['added_vec_size'] = 32
elif params['conditioning'] == 'gesture':
   params['added_vec_size'] = 16
else:
   raise ValueError()


runid = 'hetgk0hp'
taskId = 2
subject = 'C'
repetition = 5

past_count = params['past_count']
future_count = params['future_count']

transform = transforms.Compose([
    transforms.ToTensor()
])
df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_test = df[(df['Subject']=='C') & (df['Repetition']==repetition)].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,1,transform,sample_rate=jigsaws_sample_rate),
                        JigsawsGestureDataset(df_test,config,1,sample_rate=jigsaws_sample_rate),
                        JigsawsKinematicsDataset(df_test,config,1,sample_rate=jigsaws_sample_rate))

models = load_models(runid, params, device)

frames_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/ModelTesting/{runid}_{taskId}_{subject}_{repetition}_{params['conditioning']}'
os.makedirs(frames_dir, exist_ok=True)



batch_size = 1
# set models to eval
for model in models:
   model.eval()
   model.batch_size = batch_size

frame_encoder, frame_decoder, prior_lstm, generation_lstm = models

prior_lstm.batch_size = batch_size
prior_lstm.hidden = prior_lstm.init_hidden()
generation_lstm.batch_size = batch_size
generation_lstm.hidden = generation_lstm.init_hidden()

for t in tqdm(range(len(dataset_test))):

   batch = [b.unsqueeze(0) for b in dataset_test[t]]   
   
   frames, gestures, gestures_onehot, positions, rotations, kinematics, _ = unpack_batch(params, config, batch, device)           

   # keep loading past frames (for conditioning), once conditioning is over, load previously encoded frames
   if t <= params['past_count']:          
      frames_t_minus_one, skips = frame_encoder(frames[:,0,:,:,:])
   else:          
      frames_t_minus_one = frame_encoder(decoded_frames.to(device))[0]       
   
   z,mu,logvar = prior_lstm(frames_t_minus_one)        

   # load condition data of current frame
   conditioning_vec = kinematics[:,0,:]   

   # predict next frame latent, decode next frame, store next frame
   frames_to_decode = generation_lstm(torch.cat([frames_t_minus_one,mu,conditioning_vec],dim=-1).float())
   decoded_frames = frame_decoder([frames_to_decode,skips]).cpu()

   fig, axes = plt.subplots(1,2)
   
   axes[0].imshow(torch_to_numpy(decoded_frames[0,:,:,:].detach()))
   axes[1].imshow(torch_to_numpy(frames[0,0,:,:,:].detach()))
   
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

   