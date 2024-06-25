import sys
sys.path.append('/home/chen/MScProject')
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')

from Code.rarp50.rarp50KinematicsDataset import rarp50KinematicsDataset
from Code.rarp50.rarp50ImageDataset import rarp50ImageDataset
from models.blobReconstructor import BlobReconstructor, BlobConfig
import pandas as pd
import os
from DataUtils import ConcatDataset

from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import random
import numpy as np

from rarp50.rarp50Config import config

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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

params = {
   'frame_size':64,
   'batch_size': 16,
   'num_epochs':300,
   'img_compressed_size': 256,
   'prior_size': 32,      
   'lr':0.0001,
   'gamma':100,
   'conditioning':'position',
   'dataset':'rarp50'
}


transform = transforms.Compose([
    transforms.ToTensor()
])

df = pd.read_csv(os.path.join('/home/chen/MScProject/rarp50_filtered_data_detailed.csv'))
df = df[(df['videoName']=='video_37')].reset_index(drop=True)

dataset_train = ConcatDataset(rarp50ImageDataset(df,config,1),rarp50KinematicsDataset(df,config,1))
dataloader_train = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

dataset_test = ConcatDataset(rarp50ImageDataset(df,config,1),rarp50KinematicsDataset(df,config,1))
dataloader_test = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

mse = torch.nn.MSELoss()

# start_x: float,
#     start_y: int,
#     start_s: int,
#     a_range: list[int],
#     start_theta: int,
#     side: str
blobs = [
    BlobConfig(0.25,0,4,[1,5],0,'right'),
    BlobConfig(-0.25,0,4,[1,5],0,'left')
]

model = BlobReconstructor(256,blobs,params['batch_size'],include_ecm=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

models_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs/2_blobs_seed_{seed}_models'    
os.makedirs(models_dir, exist_ok=True)

images_dir = f'/home/chen/MScProject/Code/experiments/rarp50/Blobs/2_blobs_seed_{seed}_images'
os.makedirs(images_dir, exist_ok=True)

base_frame = torch.randn(dataset_test[0][0].size()).to(device)

best_valid_loss = 9999999

for epoch in (range(params['num_epochs'])):
    model.eval() # disable batch-norm
    Loss_train = 0.0
    for batch in tqdm(dataloader_train):
        frames = batch[0].to(device)
        psm1_position, psm1_rotation = batch[1][0].to(device)*100, batch[1][1].to(device)
        psm2_position, psm2_rotation = batch[1][2].to(device)*100, batch[1][3].to(device)
        ecm_position, ecm_rotation = batch[1][4].to(device)*100, batch[1][5].to(device)
        
        kinematics = torch.cat([psm1_position, psm1_rotation, psm2_position, psm2_rotation],dim=-1)
        ecm_kinematics = torch.cat([ecm_position, ecm_rotation],dim=-1)

        batch_size = frames.size(0)

        optimizer.zero_grad()        
        output_high, output_low, blobs = model(base_frame.repeat(batch_size,1,1,1),kinematics,False,ecm_kinematics)

        MSE_Loss = mse(output_low, frames)
        PER_Loss = loss_fn_vgg(output_high, frames).mean()
        Loss = params['gamma']*MSE_Loss
        Loss_train += Loss.item()
        Loss.backward()
        optimizer.step()    

    Loss_test = 0.0
    model.eval()
    with torch.no_grad():        
        for batch in tqdm(dataloader_test):
            frames = batch[0].to(device)
            psm1_position, psm1_rotation = batch[1][0].to(device)*100, batch[1][1].to(device)
            psm2_position, psm2_rotation = batch[1][2].to(device)*100, batch[1][3].to(device)
            ecm_position, ecm_rotation = batch[1][4].to(device)*100, batch[1][5].to(device)
            kinematics = torch.cat([psm1_position, psm1_rotation, psm2_position, psm2_rotation],dim=-1)
            ecm_kinematics = torch.cat([ecm_position, ecm_rotation],dim=-1)

            batch_size = frames.size(0)            
            output_high, output_low, blobs = model(base_frame.repeat(batch_size,1,1,1),kinematics,False,ecm_kinematics)

            MSE_Loss = mse(output_low, frames)
            PER_Loss = loss_fn_vgg(output_high, frames).mean()
            Loss = params['gamma']*MSE_Loss
            Loss_test += Loss.item()

    valid_loss = Loss_test/len(dataloader_test)
    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(models_dir,f"model_{epoch}.pth"))
        torch.save(model.positions_to_blobs.state_dict(),os.path.join(models_dir,f"positions_to_blobs_{epoch}.pth"))
        print('new best model')


    print(Loss_train/len(dataloader_train))
    print(Loss_test/len(dataloader_test))
    print(epoch)
    plt.figure()
    plt.subplot(3,2,1)
    plt.imshow(torch_to_numpy(output_high[0].detach().cpu()))
    plt.subplot(3,2,2)
    plt.imshow(torch_to_numpy(frames[0].detach().cpu()))

    if len(blobs)==2:
        plt.subplot(3,2,3)
        plt.imshow(torch_to_numpy(blobs[1][0].detach().cpu()))        
        plt.subplot(3,2,4)
        plt.imshow(torch_to_numpy(blobs[0][0].detach().cpu())) 
    else:
        plt.subplot(3,2,3)
        plt.imshow(torch_to_numpy(blobs[1][0].detach().cpu()+blobs[3][0].detach().cpu()))        
        plt.subplot(3,2,4)
        plt.imshow(torch_to_numpy(blobs[0][0].detach().cpu()+blobs[2][0].detach().cpu()))        

    plt.subplot(3,2,5)
    plt.imshow(torch_to_numpy(output_low[0].detach().cpu()))    

    plt.subplot(3,2,6)
    plt.imshow(torch_to_numpy(frames[0].detach().cpu()))
    plt.imshow(torch_to_numpy(blobs[1][0].detach().cpu()) + torch_to_numpy(blobs[0][0].detach().cpu()), cmap='jet', alpha=0.15)



    plt.savefig(os.path.join(images_dir,f'test_{epoch}.png'))
    plt.close()




    
    