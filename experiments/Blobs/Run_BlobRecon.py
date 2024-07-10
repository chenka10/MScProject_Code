import sys
sys.path.append('/home/chen/MScProject')
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')

from models.blobReconstructor import BlobReconstructor, BlobConfig
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

params = {
   'frame_size':64,
   'batch_size': 64,
   'num_epochs':15,
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

subject_to_leave = 'C'

df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject'] != subject_to_leave)].reset_index(drop=True)
df_test = df[(df['Subject'] == subject_to_leave)].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_train = ConcatDataset(JigsawsImageDataset(df_train,config,1,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_train,config,1,sample_rate=jigsaws_sample_rate))
dataloader_train = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,1,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_test,config,1,sample_rate=jigsaws_sample_rate))
dataloader_test = DataLoader(dataset_test,params['batch_size'],True,drop_last=True)

position_indices = config.kinematic_slave_position_indexes
rotation_indices = config.kinematic_slave_rotation_indexes

mse = torch.nn.MSELoss()


blobs = [
    BlobConfig(0.25,0,4,[2,5],-torch.pi/7,'right'),
    BlobConfig(-0.25,0,4,[2,5],0,'left')
]

model = BlobReconstructor(256,blobs,params['batch_size']).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

models_dir = f'/home/chen/MScProject/Code/experiments/Blobs/2_blobs_seed_{seed}_leave_{subject_to_leave}_models'    
os.makedirs(models_dir, exist_ok=True)

images_dir = f'/home/chen/MScProject/Code/experiments/Blobs/2_blobs_seed_{seed}_leave_{subject_to_leave}_images'
os.makedirs(images_dir, exist_ok=True)

base_frame = torch.randn(dataset_test[0][0][0].size()).to(device)

best_valid_loss = 9999999

for epoch in (range(params['num_epochs'])):
    model.eval() # disable batch-norm
    Loss_train = 0.0
    for batch in tqdm(dataloader_train):
        frames = batch[0].squeeze(1).to(device)
        positions = batch[1].squeeze(1)[:,position_indices].to(device)*100
        rotations = batch[1].squeeze(1)[:,rotation_indices].to(device)
        kinematics = torch.cat([positions[:,:3], rotations[:,:9], positions[:,3:], rotations[:,9:]],-1)

        batch_size = frames.size(0)

        optimizer.zero_grad()

        include_grippers = epoch>5
        output_high, output_low, blobs = model(base_frame.repeat(batch_size,1,1,1),kinematics,include_grippers)

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
            frames = batch[0].squeeze(1).to(device)
            positions = batch[1].squeeze(1)[:,position_indices].to(device)*100
            rotations = batch[1].squeeze(1)[:,rotation_indices].to(device)
            kinematics = torch.cat([positions[:,:3], rotations[:,:9], positions[:,3:], rotations[:,9:]],-1)
            batch_size = frames.size(0)

            include_grippers = epoch>5
            output_high, output_low, blobs = model(base_frame.repeat(batch_size,1,1,1),kinematics,include_grippers)

            MSE_Loss = mse(output_low, frames)
            PER_Loss = loss_fn_vgg(output_high, frames).mean()
            Loss = params['gamma']*MSE_Loss
            Loss_test += Loss.item()

    valid_loss = Loss_test/len(dataloader_test)
    if best_valid_loss > valid_loss:    
        print('new best model')

    best_valid_loss = valid_loss
    torch.save(model.state_dict(), os.path.join(models_dir,f"model_{epoch}.pth"))
    torch.save(model.positions_to_blobs.state_dict(),os.path.join(models_dir,f"positions_to_blobs_{epoch}.pth"))


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




    
    