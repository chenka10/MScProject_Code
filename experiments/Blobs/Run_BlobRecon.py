import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')

from models.blobReconstructor import BlobReconstructor
import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import random
import numpy as np

from Jigsaws.JigsawsConfig import main_config as config

import matplotlib.pyplot as plt

import lpips


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('seed:', seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

params = {
   'frame_size':64,
   'batch_size': 8,
   'num_epochs':250,
   'img_compressed_size': 256,
   'prior_size': 32,   
   'past_count': 10,
   'future_count': 30,
   'num_gestures': 16, 
   'lr':0.0001,
   'gamma':100,
   'conditioning':'position', #'gesture'   
   'dataset':'JIGSAWS'
}

past_count = params['past_count']
future_count = params['future_count']

transform = transforms.Compose([
    transforms.ToTensor()
])
df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject'] == 'C') & (df['Repetition'] != 1)].reset_index(drop=True)
df_test = df[(df['Subject']=='C') & (df['Repetition'] == 1)].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_train = ConcatDataset(JigsawsImageDataset(df_train,config,1,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_train,config,1,sample_rate=jigsaws_sample_rate))
dataloader_train = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,1,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_test,config,1,sample_rate=jigsaws_sample_rate))
dataloader_test = DataLoader(dataset_test,params['batch_size'],True,drop_last=True)

position_indices = config.kinematic_slave_position_indexes

mse = torch.nn.MSELoss()

model = BlobReconstructor(256,3,params['batch_size']).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

base_frame = dataset_test[0][0][0].to(device)

for epoch in (range(params['num_epochs'])):
    model.train()
    Loss_train = 0.0
    for batch in tqdm(dataloader_train):
        frames = batch[0].squeeze(1).to(device)
        positions = batch[1].squeeze(1)[:,position_indices].to(device)

        batch_size = frames.size(0)

        optimizer.zero_grad()

        recon_frames, blobs1, blobs2 = model(base_frame.repeat(batch_size,1,1,1),positions)

        MSE_Loss = mse(recon_frames, frames)
        PER_Loss = loss_fn_vgg(recon_frames, frames).mean()
        Loss = PER_Loss + params['gamma']*MSE_Loss
        Loss_train += Loss.item()
        Loss.backward()
        optimizer.step()

    Loss_test = 0.0
    with torch.no_grad():        
        for batch in tqdm(dataloader_test):
            frames = batch[0].squeeze(1).to(device)
            positions = batch[1].squeeze(1)[:,position_indices].to(device)

            batch_size = frames.size(0)

            recon_frames, blobs1, blobs2 = model(base_frame.repeat(batch_size,1,1,1),positions)

            MSE_Loss = mse(recon_frames, frames)
            PER_Loss = loss_fn_vgg(recon_frames, frames).mean()
            Loss = PER_Loss + params['gamma']*MSE_Loss
            Loss_test += Loss.item()


    print(Loss_train/len(dataloader_train))
    print(Loss_test/len(dataloader_test))
    print(epoch)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(torch_to_numpy(recon_frames[0].detach().cpu()))
    plt.subplot(2,2,2)
    plt.imshow(torch_to_numpy(frames[0].detach().cpu()))
    plt.subplot(2,2,3)
    plt.imshow(torch_to_numpy(blobs1[0].detach().cpu()),vmin=0,vmax=1)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(torch_to_numpy(blobs2[0].detach().cpu()),vmin=0,vmax=1)    
    plt.colorbar()
    plt.savefig(f'test_{epoch}.png')
    plt.close()




    
    