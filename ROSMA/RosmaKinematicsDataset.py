from ROSMA.RosmaDatasetBase import RosmaDatasetBase
import torch

import torchvision.transforms as T
from ROSMA.RosmaUtils import get_kinematics


class RosmaKinematicsDataset(RosmaDatasetBase):
  def __init__(self, data_csv_df, config, frames_to_retrieve,sample_rate):
      super().__init__(data_csv_df, config, frames_to_retrieve,sample_rate)        

  def __len__(self):
    return super().__len__()

  def __getitem__(self, index):        
    _, kinematic_row_orig, task_id, subject, repetition = super().get_frame_indexes_and_rosma_meta_by_item_index(index)

    kinematics = []

    for i in range(self.frames_to_retrieve):
      kinematics.append(get_kinematics(self.config, kinematic_row_orig, task_id, subject, repetition).tolist()[1:])
    
    return  torch.tensor(kinematics)
    
