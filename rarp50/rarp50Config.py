import os

class rarp50Config():
  def __init__(self):    
    self.project_baseDir = '/home/chen/MScProject/'
    self.rarp50_videoFramesDir = os.path.join(self.project_baseDir,'data/rarp50')    
    self.rarp50_kinematicsDir = os.path.join(self.project_baseDir,'data/rarp50_kinematics')    

config = rarp50Config()