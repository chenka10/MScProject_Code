import os
from torch.utils.data import Dataset
import numpy as np
from Code.rarp50.rarp50DatasetBase import rarp50DatasetBase
from PIL import Image
import torchvision.transforms as T

class rarp50ImageDataset(rarp50DatasetBase):
  def __init__(self,data_csv_df,config,frames_to_retrieve):
     super().__init__(data_csv_df,config,frames_to_retrieve)    

  def __getitem__(self, index):      
      video_frame_index, video_index = super().get_frame_index_by_dataset_index(index)
      video_name = self.df.at[video_index,'videoName']
      frame_path = os.path.join(self.config.rarp50_videoFramesDir,video_name,f'{video_frame_index}.jpg')
      
      frame = np.array(Image.open(frame_path))
      transform = T.Compose([T.ToTensor()])
      frame = transform(frame)

      return frame
      