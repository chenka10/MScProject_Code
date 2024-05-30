import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/models')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')

from models.blobReconstructor import BlobConfig, BlobReconstructor
import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from torchvision.transforms import transforms
import torch

blobs = [
    BlobConfig(0.25,0,4,[2,4],'right'),
    BlobConfig(-0.25,0,4,[2,4],'left')
]

models_dir = f'/home/chen/MScProject/Code/experiments/Blobs/seed_3_42_models' 

model = BlobReconstructor(256,blobs,16)
model.load_state_dict(torch.load(os.path.join(models_dir,'model_5.pth')))
model.eval()  # Set the model to evaluation mode

positions_to_blobs = model.positions_to_blobs

torch.save(positions_to_blobs.state_dict(), os.path.join(models_dir,'positions_to_blobs.pth'))




