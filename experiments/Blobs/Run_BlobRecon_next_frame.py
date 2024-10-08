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
   'batch_size': 64,
   'num_epochs':250,
   'img_compressed_size': 256,
   'prior_size': 32,   
   'past_count': 20,
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

base_frame = dataset_test[0][0][0].to(device)

best_valid_loss = 9999999

for epoch in (range(params['num_epochs'])):
    model.eval() # disable batch-norm
    Loss_train = 0.0
    for batch in tqdm(dataloader_train):
        frames_tm1 = batch[0][:,0].to(device) # tm1 - t minus 1
        frames_t = batch[0][:,1].to(device)
        positions_tm1 = batch[1][:,0][:,position_indices].to(device)
        positions_t = batch[1][:,1][:,position_indices].to(device)

        batch_size = frames_t.size(0)

        optimizer.zero_grad()

        output_high, output_low, blobs = model(frames_tm1,positions_t)

        d = get_distance(positions_t, positions_tm1)
        MSE_Loss = mse(output_low, frames_t).mean(-1).mean(-1).mean(-1)
        PER_Loss = loss_fn_vgg(output_high, frames_t).mean(-1).mean(-1).mean(-1)
        Loss = params['gamma']*MSE_Loss + PER_Loss
        Loss = (Loss*d).mean()
        Loss_train += Loss.item()
        Loss.backward()
        optimizer.step()    

    Loss_test = 0.0
    model.eval()
    with torch.no_grad():        
        for batch in tqdm(dataloader_test):
            frames_tm1 = batch[0][:,0].to(device) # tm1 - t minus 1
            frames_t = batch[0][:,1].to(device)
            positions_tm1 = batch[1][:,0][:,position_indices].to(device)
            positions_t = batch[1][:,1][:,position_indices].to(device)

            batch_size = frames_t.size(0)

            output_high, output_low, blobs = model(frames_tm1,positions_t)

            d = get_distance(positions_t, positions_tm1)
            MSE_Loss = mse(output_low, frames_t).mean(-1).mean(-1).mean(-1)
            PER_Loss = loss_fn_vgg(output_high, frames_t).mean(-1).mean(-1).mean(-1)
            Loss = params['gamma']*MSE_Loss + PER_Loss
            Loss = (Loss*d).mean()
            Loss_test += Loss.item()

    valid_loss = Loss_test/len(dataloader_test)
    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(models_dir,f"model_{epoch}.pth"))
        print('new best model')


    print(Loss_train/len(dataloader_train))
    print(Loss_test/len(dataloader_test))
    print(epoch)
    plt.figure()
    plt.subplot(3,2,1)
    plt.imshow(torch_to_numpy(output_high[0].detach().cpu()))
    plt.subplot(3,2,2)
    plt.imshow(torch_to_numpy(frames_t[0].detach().cpu()))
    plt.subplot(3,2,3)
    plt.imshow(torch_to_numpy(blobs[1][0].detach().cpu()))        
    plt.subplot(3,2,4)
    plt.imshow(torch_to_numpy(blobs[0][0].detach().cpu()))        
    plt.subplot(3,2,5)
    plt.imshow(torch_to_numpy(output_low[0].detach().cpu()))               
    plt.savefig(os.path.join(images_dir,f'test_{epoch}.png'))
    plt.close()




    
    