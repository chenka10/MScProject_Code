import os
from torch.utils.data import Dataset
import numpy as np
from Code.rarp50.rarp50DatasetBase import rarp50DatasetBase
from PIL import Image
import torchvision.transforms as T

# in rarp50 segmentations:
# 0 - background
# 1 - claw "fingers"
# 2 - claw "base"
# 3 - arm
# 4 - needle
# 5 - string
rarp50_segmentation_map = [0,255,255,255,0,0,0,0,0,0,0]

class rarp50ImageDataset(rarp50DatasetBase):
  def __init__(self,data_csv_df,config,frames_to_retrieve,digits_in_name = 5, frame_increments = 12, is_segmentations = False):
     super().__init__(data_csv_df,config,frames_to_retrieve, frame_increments)    
     self.digits_in_name = digits_in_name
     self.is_segmentations = is_segmentations

  def __getitem__(self, index):      
      video_frame_index, video_index = super().get_videoStartFrame_and_videoIndex_by_dataset_index(index)
      video_name = self.df.at[video_index,'videoName']

      frame_path = os.path.join(self.config.rarp50_videoFramesDir,video_name,f'{str(video_frame_index).zfill(self.digits_in_name)}.jpg')

      if os.path.exists(frame_path) is False:
         frame_path = os.path.join(self.config.rarp50_videoFramesDir,video_name,f'{str(video_frame_index).zfill(self.digits_in_name)}.png')         
      
      frame = np.array(Image.open(frame_path))

      if self.is_segmentations:
         for i in range(frame.max()+1):
            frame[frame==i] = rarp50_segmentation_map[i]

      transform = T.Compose([T.ToTensor()])
      frame = transform(frame)

      return frame
      