import sys
sys.path.append('/home/chen/MScProject/Code/')

from RosmaImageDataset import RosmaImageDataset
from RosmaKinematicsDataset import RosmaKinematicsDataset
import pandas as pd
from RosmaConfig import config
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from DataUtils import ConcatDataset

transform = transforms.Compose([
    transforms.ToTensor()
])

df = pd.read_csv('~/MScProject/rosma_all_data_detailed.csv')
df = df[(df['TaskID']==0) & (df['SubjectID']=='X01')].reset_index(drop=True)

id = RosmaImageDataset(df,config,20,transforms=transform,sample_rate=5)
kd = RosmaKinematicsDataset(df, config, 20, 5)

ds = ConcatDataset(id,kd)

dl = DataLoader(ds,8)

for d in tqdm(dl):
    a = 5




