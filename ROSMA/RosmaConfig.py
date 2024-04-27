
class RosmaStorage():
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

class RosmaConfig:
    def __init__(self):
        self.task_names = [
            'Pea_on_a_Peg',
            'Wire_Chaser_I',
            'Post_and_Sleeve'
        ]
        self.extracted_frames_dir = '/home/chen/MScProject/data/ROSMA_frames_128'
        self.kinematics_files_dir = '/home/chen/MScProject/data/ROSMA'
        self.kinematics_df_storage = RosmaStorage()
        self.project_baseDir = '/home/chen/MScProject/'

        self.kinematic_slave_position_indexes = [80,81,82,116,117,118]
        self.kinematic_slave_orientation_indexes = [83,84,85,86,119,120,121,122]

    def get_task_id(self, task_name):
        return self.task_names.index(task_name)  

    def get_task_name(self, task_id):
        return self.task_names[task_id]  

config = RosmaConfig()