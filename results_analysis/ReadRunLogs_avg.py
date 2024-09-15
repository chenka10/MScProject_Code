import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 20
NUM_SUBJECTS = len(subjects)

runid_names = {
    'B':{
        'runids':["2yhyq225","iy1t5egp"],
        'run_names':["leave B","leave B (Blobs)"]
    },
    'C':{
        'runids':["vten6hg4","1lg477ow"],
        'run_names':["leave C","leave C (Blobs)"]
    },
    'D':{
        'runids':["i91wiw8v","92azggnk"],
        'run_names':["leave D","leave D (Blobs)"]
    },
    'E':{
        'runids':["6bksh0iz","adm76oiv"],
        'run_names':["leave E","leave E (Blobs)"]
    },
    'F':{
        'runids':["c7etn81h","u9e76hxl"],
        'run_names':["leave F","leave F (Blobs)"]
    },
    'G':{
        'runids':["pteyjtdc","72fplaw5"],
        'run_names':["leave G","leave G (Blobs)"]
    },
    'H':{
        'runids':["526eqa92","ay67z0qn"],
        'run_names':["leave H","leave H (Blobs)"]
    },
    'I':{
        'runids':["misunawm","72f7731x"],
        'run_names':["leave I","leave I (Blobs)"]
    }
}

epochs_to_take = 18
epoch_to_examine = 16

def get_ssim_results_from_wandb(runid):
    last_epoch_ssim_train = None
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[-1][[f'valid_SSIM_timestep_{i}' for i in range(NUM_OF_PRED_FRAMES)]]

        valid_last_ssim = hist['valid_SSIM_timestep_9'][:epochs_to_take]

        num_epochs = hist.shape[0]

    return last_epoch_ssim_valid, valid_last_ssim, num_epochs

sum_last_epoch_ssim_valid_BLOB = None
sum_last_epoch_ssim_valid_SVG = None

fig = plt.figure(figsize=(8,4))    
for subject_i,subject in enumerate(subjects):

    runids = runid_names[subject]['runids']
    run_names = runid_names[subject]['run_names']
    name = subject    

    for i,runid in enumerate(runids):

        last_epoch_ssim_valid, _, num_epochs = get_ssim_results_from_wandb(runid)  

        if sum_last_epoch_ssim_valid_BLOB is None:    
            if i==0:        
                sum_last_epoch_ssim_valid_SVG = last_epoch_ssim_valid.values            
            else:
                sum_last_epoch_ssim_valid_BLOB = last_epoch_ssim_valid.values            
        else:       
            if i==0:        
                sum_last_epoch_ssim_valid_SVG += last_epoch_ssim_valid.values            
            else:
                sum_last_epoch_ssim_valid_BLOB += last_epoch_ssim_valid.values 
                
        
        # if i==0:
        #     plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][0],alpha=1,color='tab:blue')
        # else:
        #     plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][1],alpha=1,color='tab:orange')
        plt.xlabel('Generated Frame Number')
        plt.ylabel('SSIM')        
        plt.ylim([0.7,1])
        plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        


plt.plot(sum_last_epoch_ssim_valid_SVG/NUM_SUBJECTS, marker = 'x', color='tab:blue',label='mean (SVG*)')
plt.plot(sum_last_epoch_ssim_valid_BLOB/NUM_SUBJECTS, marker = 'x', color='tab:orange',label='mean (Blobs)')
plt.axvline(x=9,color='red',linestyle='--',label='last pred. during training')


# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[-2:], labels[-2:])
plt.legend()

plt.title(f'Generated Future-Frame SSIM (LOUO folds mean)')        

plt.tight_layout()


fig.savefig(f'new_SSIM_results_all_position_blob_svg_avg1.png')
plt.close()


