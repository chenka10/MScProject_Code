import sys
sys.path.append('/home/chen/MScProject/')
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

from Code.rarp50.rarp50KinematicsDataset import rarp50KinematicsDataset
from rarp50Config import config
import pandas as pd

df = pd.read_csv('/home/chen/MScProject/rarp50_filtered_data_detailed.csv')
kds = rarp50KinematicsDataset(df,config,10)
kds[8]




