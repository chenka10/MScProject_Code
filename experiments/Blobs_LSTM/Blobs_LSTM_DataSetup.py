from Jigsaws.JigsawsUtils import jigsaws_to_quaternions
from utils import expand_positions
from torchvision.transforms import transforms
from DataUtils import ConcatDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsGestureDataset import JigsawsGestureDataset
from ROSMA.RosmaImageDataset import RosmaImageDataset
from ROSMA.RosmaKinematicsDataset import RosmaKinematicsDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch

def get_dataloaders(params, config):

    # Define dataset and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if params['dataset'] == 'JIGSAWS':

        jigsaws_sample_rate = 6 # this is not [Hz], this is sampling 1 ouf of 6 frames from the frames we have sampled at 30[Hz]

        df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
        df_train = df[(df['Subject']=='C') & df['Repetition']==1].reset_index(drop=True)
        df_valid = df[(df['Subject']=='C') & df['Repetition']==1].reset_index(drop=True)

        dataset_train = ConcatDataset(JigsawsImageDataset(df_train,config,params['past_count']+params['future_count'],transform,sample_rate=jigsaws_sample_rate),
                                JigsawsGestureDataset(df_train,config,params['past_count']+params['future_count'],sample_rate=jigsaws_sample_rate),
                                JigsawsKinematicsDataset(df_train,config,params['past_count']+params['future_count'],sample_rate=jigsaws_sample_rate))
        dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)

        dataset_valid = ConcatDataset(JigsawsImageDataset(df_valid,config,params['past_count']+params['future_count'],transform,sample_rate=jigsaws_sample_rate),
                                JigsawsGestureDataset(df_valid,config,params['past_count']+params['future_count'],sample_rate=jigsaws_sample_rate),
                                JigsawsKinematicsDataset(df_valid,config,params['past_count']+params['future_count'],sample_rate=jigsaws_sample_rate))
        dataloader_valid = DataLoader(dataset_valid, batch_size=params['batch_size'], shuffle=True)

        params['train_subjects'] = df_train['Subject'].unique()
        params['train_repetitions'] = df_train['Repetition'].unique()
        params['valid_subjects'] = df_valid['Subject'].unique()
        params['valid_repetitions'] = df_valid['Repetition'].unique()

        if params['frame_size'] == 128:
            config.extracted_frames_dir = '/home/chen/MScProject/data/jigsaws_extracted_frames_128/'

    elif params['dataset'] == 'ROSMA':

        # note, this is the sample rate of the video frames (and kinematics) that will be used
        rosma_target_sample_rate = 5 # [Hz]

        df = pd.read_csv(os.path.join(config.project_baseDir,'rosma_all_data_detailed.csv'))
        df_train = df[(df['TaskID']==0) & (df['Subject']!='X05')].reset_index(drop=True)
        df_valid = df[(df['TaskID']==0) & (df['Subject']=='X05')].reset_index(drop=True)

        dataset_train = ConcatDataset(RosmaImageDataset(df_train,config,params['past_count']+params['future_count'],transform,sample_rate=rosma_target_sample_rate),                                
                                RosmaKinematicsDataset(df_train,config,params['past_count']+params['future_count'],sample_rate=rosma_target_sample_rate))
        dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)

        dataset_valid = ConcatDataset(RosmaImageDataset(df_valid,config,params['past_count']+params['future_count'],transform,sample_rate=rosma_target_sample_rate),                                
                                RosmaKinematicsDataset(df_valid,config,params['past_count']+params['future_count'],sample_rate=rosma_target_sample_rate))
        dataloader_valid = DataLoader(dataset_valid, batch_size=params['batch_size'], shuffle=True)

        params['train_subjects'] = df_train['Subject'].unique()
        params['train_repetitions'] = df_train['Repetition'].unique()
        params['valid_subjects'] = df_valid['Subject'].unique()
        params['valid_repetitions'] = df_valid['Repetition'].unique()


    return dataloader_train, dataloader_valid


def unpack_batch(params, config, batch, device):
    if params['dataset'] == 'JIGSAWS':

        position_indices = config.kinematic_slave_position_indexes
        rotation_indices = config.kinematic_slave_rotation_indexes

        # load batch data
        frames = batch[0].to(device)
        gestures = batch[1].detach().cpu()
        gestures_onehot = torch.nn.functional.one_hot(batch[1],params['num_gestures']).to(device)
        positions = batch[2][:,:,position_indices].to(device)    
        positions_expanded = expand_positions(positions)
        rotations = batch[2][:,:,rotation_indices].to(device) 
        rotations = jigsaws_to_quaternions(rotations) 
        kinematics = torch.concat([positions_expanded,rotations],dim=-1)
        batch_size = frames.size(0)           
    
    if params['dataset'] == 'ROSMA':

        position_indices = config.kinematic_slave_position_indexes
        rotation_indices = config.kinematic_slave_orientation_indexes

        # load batch data
        frames = batch[0].to(device)
        batch_size = frames.size(0)   

        # mock gestures
        gestures = torch.ones((batch_size,params['seq_len'])).to(torch.int64)
        gestures_onehot = torch.nn.functional.one_hot(gestures,params['num_gestures']).to(device)

        # kinematics
        positions = batch[1][:,:,position_indices].to(device)    
        positions_expanded = expand_positions(positions)
        rotations = batch[1][:,:,rotation_indices].to(device)         
        kinematics = torch.concat([positions_expanded,rotations],dim=-1)

    return (frames, gestures, gestures_onehot, positions, rotations, kinematics, batch_size) 
        

