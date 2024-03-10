from JigsawsDatasetBase import JigsawsDatasetBase
import torch
import JigsawsUtils as ju
import torchvision.transforms as T
import pdb

class JigsawsGestureDataset(JigsawsDatasetBase):
  def __init__(self, data_csv_df, config, frames_to_retrieve, sample_rate):
      super().__init__(data_csv_df, config, frames_to_retrieve, sample_rate)        

  def __len__(self):
    return super().__len__()

  def __getitem__(self, index):        
    frame_index_resolution, frame_index_orig, task_id, subject, repetition = super().get_frame_indexes_and_jigsaws_meta_by_item_index(index)

    all_gestures = ju.jigsaws_get_gestures(self.config,task_id, subject, repetition)

    gestures = []

    for i in range(self.frames_to_retrieve):
      gestures.append(all_gestures[frame_index_orig + i*self.sample_rate])
    
    return  torch.tensor(gestures)
    
