import os

class JigsawsStorage():
    def __init__(self):
        self.data = {}

    def add_entry(self, df, task_id, subject, repetition):
        key = (task_id, subject, repetition)
        self.data[key] = df

    def get_entry(self, task_id, subject, repetition):
        key = (task_id, subject, repetition)
        return self.data.get(key, None)

    def entry_exists(self, task_id, subject, repetition):
        key = (task_id, subject, repetition)
        return key in self.data

class JigsawsConfig():
  def __init__(self):
    self.task_names = ['Knot_Tying','Needle_Passing','Suturing']
    self.skill_names = ['GRS','Respect for tissue',
	  'Suture/needle handling',
	  'Time and motion',
	  'Flow of operation',
	  'Overall performance',
	  'Quality of final product']
    self.project_baseDir = '/home/chen/MScProject/'
    self.jigsaws_baseDir = os.path.join(self.project_baseDir,'jigsaws')
    self.extracted_frames_dir = '/home/chen/MScProject/data/jigsaws_extracted_frames_64/'
    self.metadata_dir = '/home/chen/MScProject/data/metadata/'
    self.kinematics_units_slices = {
        'MLT':slice(0,19),
        'MRT':slice(19,38),
        'SLT':slice(38,57),
        'SRT':slice(57,76),
        'SlaveBoth':slice(38,76)
        }
    self.kinematics_variables_slices = {
       'xyz':slice(0,3),
       'R':slice(3,12),
       'trans_vel':slice(12,15),
       'rot_vel':slice(15,18),
       'gripper_angle':slice(18,19),
       'both':slice(0,39)
    }
    self.kinematic_slave_position_indexes = [38,39,40,57,58,59]
    self.kinematic_master_position_indexes = [0, 1, 2, 19, 20, 21]
    self.kinematics_df_storage = JigsawsStorage()
    self.gestures_storage = JigsawsStorage()

  def get_project_dir(self):
    return self.project_baseDir

  def get_task_name(self, task_id):
    return self.task_names[task_id]

  def get_task_id(self, task_name):
    return self.task_names.index(task_name)
  
  def get_metadata_dir(self):
     return self.metadata_dir

  def get_kinematic_unit_slices(self, unit_name):
    return self.kinematics_units_slices[unit_name]

  def get_kinematic_variable_slices(self, kinematic_variable_name):
    return self.kinematics_variables_slices[kinematic_variable_name]

  def get_kinematic_slices(self, kinematic_unit_name, kinematic_variable_name):
    ku_slice = self.kinematics_units_slices[kinematic_unit_name]
    kv_slice = self.kinematics_variables_slices[kinematic_variable_name]

    ret = slice(ku_slice.start+kv_slice.start,ku_slice.start+kv_slice.stop)
    return ret

main_config = JigsawsConfig()