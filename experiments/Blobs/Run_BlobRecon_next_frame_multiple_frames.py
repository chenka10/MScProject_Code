import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')

from models.blobReconstructorNextFrame import BlobReconstructorNextFrame, BlobConfig
import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy, get_distance
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
   'batch_size': 16,
   'num_epochs':250,
   'img_compressed_size': 256,
   'prior_size': 32,   
   'past_count': 10,
   'future_count': 0,
   'num_gestures': 16, 
   'lr':0.0001,
   'gamma':100,
   'conditioning':'position', #'gesture'   
   'dataset':'JIGSAWS'
}

past_count = params['past_count']
future_count = params['future_count']
seq_len = past_count+future_count

transform = transforms.Compose([
    transforms.ToTensor()
])
df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_train = df[(df['Subject'] != 'C')].reset_index(drop=True)
df_test = df[(df['Subject'] =='C')].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_train = ConcatDataset(JigsawsImageDataset(df_train,config,seq_len,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_train,config,seq_len,sample_rate=jigsaws_sample_rate))
dataloader_train = DataLoader(dataset_train,params['batch_size'],True,drop_last=True)

dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,seq_len,transform,sample_rate=jigsaws_sample_rate),                        
                        JigsawsKinematicsDataset(df_test,config,seq_len,sample_rate=jigsaws_sample_rate))
dataloader_test = DataLoader(dataset_test,params['batch_size'],True,drop_last=True)

position_indices = config.kinematic_slave_position_indexes

mse = torch.nn.MSELoss(reduce=False)


blobs = [
    BlobConfig(0.25,0,4,[2,4],'right'),
    BlobConfig(-0.25,0,4,[2,4],'left')
]

model = BlobReconstructorNextFrame(256,blobs,params['batch_size']).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

models_dir = f'/home/chen/MScProject/Code/experiments/Blobs_next_frame/seed_{seed}_models'    
os.makedirs(models_dir, exist_ok=True)

images_dir = f'/home/chen/MScProject/Code/experiments/Blobs_next_frame/seed_{seed}_images'
os.makedirs(images_dir, exist_ok=True)

best_valid_loss = 9999999

mover_index = -1

for epoch in (range(params['num_epochs'])):
    model.eval() # disable batch-norm
    Loss_train = 0.0
    for batch in tqdm(dataloader_train):        
        frames = batch[0].to(device)
        positions = batch[1][:,:,position_indices].to(device)

        batch_size = frames.size(0)
        optimizer.zero_grad()

        predicted_sequence_high = torch.zeros([batch_size,seq_len-1,3,params['frame_size'],params['frame_size']]).to(device)        
        predicted_sequence_low = torch.zeros([batch_size,seq_len-1,3,params['frame_size'],params['frame_size']]).to(device)

        for i in range(1,seq_len):
            
            if i==1:
                prev_frame = frames[:,0,:,:,:]
            else:
                prev_frame = output_high

            target_frame = frames[:,i,:,:,:]
            target_position = positions[:,i,:]

            output_high, output_low, blobs = model(prev_frame,target_position)
            predicted_sequence_high[:,i-1,:,:,:] = output_high
            predicted_sequence_low[:,i-1,:,:,:] = output_low

        distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:]).mean(1)
        MSE_Loss = mse(predicted_sequence_low, frames[:,1:,:,:,:]).mean(-1).mean(-1).mean(-1)
        PER_Loss = loss_fn_vgg(
            predicted_sequence_high.reshape([batch_size*(seq_len-1),3,params['frame_size'],params['frame_size']]),
            frames[:,1:,:,:,:].reshape([batch_size*(seq_len-1),3,params['frame_size'],params['frame_size']])
            ).squeeze(-1).squeeze(-1).squeeze(-1).reshape([batch_size,seq_len-1])    
        Loss = params['gamma']*MSE_Loss + PER_Loss
        Loss = (Loss.mean(1)*distance_weight).mean()
        Loss_train += Loss.item()
        Loss.backward()
        optimizer.step()    

    Loss_test = 0.0
    model.eval()
    with torch.no_grad():        
        for batch in tqdm(dataloader_test):
            frames = batch[0].to(device)
            positions = batch[1][:,:,position_indices].to(device)

            batch_size = frames.size(0)
            optimizer.zero_grad()

            predicted_sequence_high = torch.zeros([batch_size,seq_len-1,3,params['frame_size'],params['frame_size']]).to(device)        
            predicted_sequence_low = torch.zeros([batch_size,seq_len-1,3,params['frame_size'],params['frame_size']]).to(device)

            for i in range(1,seq_len):
                
                if i==1:
                    prev_frame = frames[:,0,:,:,:]
                else:
                    prev_frame = output_high

                target_frame = frames[:,i,:,:,:]
                target_position = positions[:,i,:]

                output_high, output_low, blobs = model(prev_frame,target_position)
                predicted_sequence_high[:,i-1,:,:,:] = output_high
                predicted_sequence_low[:,i-1,:,:,:] = output_low

            distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:]).mean(1)
            mover_index = distance_weight.argmax().item()
            MSE_Loss = mse(predicted_sequence_low, frames[:,1:,:,:,:]).mean(-1).mean(-1).mean(-1)
            PER_Loss = loss_fn_vgg(
                predicted_sequence_high.reshape([batch_size*(seq_len-1),3,params['frame_size'],params['frame_size']]),
                frames[:,1:,:,:,:].reshape([batch_size*(seq_len-1),3,params['frame_size'],params['frame_size']])
                ).squeeze(-1).squeeze(-1).squeeze(-1).reshape([batch_size,seq_len-1])    
            Loss = params['gamma']*MSE_Loss + PER_Loss
            Loss = (Loss.mean(1)*distance_weight).mean()
            Loss_test += Loss.item()

    valid_loss = Loss_test/len(dataloader_test)
    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(models_dir,f"model_{epoch}.pth"))
        print('new best model')


    print(Loss_train/len(dataloader_train))
    print(Loss_test/len(dataloader_test))
    print(epoch)
    plt.figure(figsize=(16,3))
    for i in range(seq_len-1):
        plt.subplot(3,seq_len-1,i+1)
        plt.imshow(torch_to_numpy(predicted_sequence_high[mover_index,i,:,:,:].detach().cpu()))
        plt.axis('off')
        plt.subplot(3,seq_len-1,(seq_len-1)+i+1)
        plt.imshow(torch_to_numpy(predicted_sequence_low[mover_index,i,:,:,:].detach().cpu()))    
        plt.axis('off')
        plt.subplot(3,seq_len-1,2*(seq_len-1)+i+1)
        plt.imshow(torch_to_numpy(frames[mover_index,i,:,:,:].detach().cpu()))    
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir,f'test_{epoch}.png'))
    plt.close()




    
    