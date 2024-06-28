import os
import torch
from torch.utils.data import Dataset
import numpy as np
from Code.rarp50.rarp50DatasetBase import rarp50DatasetBase
from PIL import Image
import torchvision.transforms as T
import pandas as pd

class rarp50KinematicsDataset(rarp50DatasetBase):
   def __init__(self,data_csv_df,config,frames_to_retrieve, frame_increments):
      super().__init__(data_csv_df,config,frames_to_retrieve, frame_increments) 
      self.storage = {}


   def kinematics_tensor_to_position_rotation(self, tensor):      
      position = torch.tensor([tensor[3],tensor[7],tensor[11]])
      rotation = torch.tensor([tensor[0],tensor[1],tensor[2],tensor[4],tensor[5],tensor[6],tensor[8],tensor[9],tensor[10]])      

      return position,rotation

   def __getitem__(self, index):  
      video_frame_index, video_index = super().get_videoStartFrame_and_videoIndex_by_dataset_index(index)
      video_name = self.df.at[video_index,'videoName']
      kinematics_file_path = os.path.join(self.config.rarp50_kinematicsDir,video_name,'DaVinciSiMemory_sync.csv')

      df = None

      if kinematics_file_path in self.storage:
         df = self.storage[kinematics_file_path]
      else:
         df = pd.read_csv(kinematics_file_path)
         self.storage[kinematics_file_path] = df

      psm1_positions, psm1_rotations, psm2_positions, psm2_rotations, ecm_positions, ecm_rotations = [],[],[],[],[],[]

      for frame_i in range(self.frames_to_retrieve):

         ecm_kinematics_string = df.at[video_frame_index,'data.Pose_ECM']
         float_list = [float(x) for x in ecm_kinematics_string.split()]
         tensor = torch.tensor(float_list)
         ecm_position, ecm_rotation = self.kinematics_tensor_to_position_rotation(tensor)

         psms_kinematic_string = df.at[video_frame_index,'data.Pose_PSM']
         float_list = [float(x) for x in psms_kinematic_string.split()]
         psm1_tensor = torch.tensor(float_list[:12])
         psm2_tensor = torch.tensor(float_list[12:24])
         psm1_position,psm1_rotation = self.kinematics_tensor_to_position_rotation(psm1_tensor)
         psm2_position,psm2_rotation = self.kinematics_tensor_to_position_rotation(psm2_tensor)

         psm1_positions.append(psm1_position)
         psm1_rotations.append(psm1_rotation)
         psm2_positions.append(psm2_position)
         psm2_rotations.append(psm2_rotation)
         ecm_positions.append(ecm_position)
         ecm_rotations.append(ecm_rotation)

         video_frame_index += self.frame_increments

      return torch.vstack(psm1_positions),\
            torch.vstack(psm1_rotations),\
            torch.vstack(psm2_positions),\
            torch.vstack(psm2_rotations),\
            torch.vstack(ecm_positions),\
            torch.vstack(ecm_rotations)