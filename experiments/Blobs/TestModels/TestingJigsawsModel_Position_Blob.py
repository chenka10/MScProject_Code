import sys
sys.path.append('/home/chen/MScProject')
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')
sys.path.append('/home/chen/MScProject/Code/experiments/Blobs')

import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from Jigsaws.JigsawsGestureDataset import JigsawsGestureDataset
from torchvision.transforms import transforms
from models.blobReconstructor import BlobReconstructor, BlobConfig
import torch
from utils import torch_to_numpy
from modelUtils import load_models, iterate_on_images
from experiments.LSTM_AutoEncoder_Clean.DataSetup import unpack_batch
from tqdm import tqdm
import gc
import time

import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

params = {
   'frame_size':64,   
   'batch_size':1,
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
repetition = 1


for repetition in [1,2,3,4]:

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

   df_for_test_frame = df[(df['Subject'] =='C')].reset_index(drop=True)
   dataset_for_test_frame = ConcatDataset(JigsawsImageDataset(df_test,config,1,transform,sample_rate=jigsaws_sample_rate),                        
                           JigsawsKinematicsDataset(df_test,config,1,sample_rate=jigsaws_sample_rate))

   model_path = '/home/chen/MScProject/Code/experiments/Blobs/2_blobs_seed_42_leave_C_models/model_14.pth'
   blobs = [
    BlobConfig(0.25,0,4,[2,5],-torch.pi/7,'right'),
    BlobConfig(-0.25,0,4,[2,5],0,'left')
   ]
   model = BlobReconstructor(256,blobs,1).to(device)
   model.load_state_dict(torch.load(model_path))
   model.eval()

   frames_dir = f'/home/chen/MScProject/Code/experiments/Blobs/ModelTesting/1_{subject}_{repetition}/'
   os.makedirs(frames_dir, exist_ok=True)



   batch_size = 1
   base_frame = dataset_for_test_frame[0][0][0].to(device)

   for t in tqdm(range(len(dataset_test))):

      batch = [b.unsqueeze(0) for b in dataset_test[t]]   
      
      frames, gestures, gestures_onehot, positions, rotations, kinematics, _ = unpack_batch(params, config, batch, device)           
      kinematics = torch.cat([positions[:,:,:3]*100, rotations[:,:,:9], positions[:,:,3:]*100, rotations[:,:,9:]],-1)

      generated_frames,_, blobs = model(base_frame.repeat(batch_size,1,1,1),kinematics[:,0,:],include_gripper=False)

      fig = plt.figure()      
      
      plt.imshow(torch_to_numpy(frames[0, 0,:,:,:].detach()))
      plt.imshow(torch_to_numpy(blobs[0][0,:,:,:].detach()) + torch_to_numpy(blobs[1][0,:,:,:].detach()),alpha=0.15)
      
      
      plt.tight_layout()
      plt.savefig(os.path.join(frames_dir,f'test_{t}.png'))
      plt.close()   

      gc.collect()   
      torch.cuda.empty_cache()

      