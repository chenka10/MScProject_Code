import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

# subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 20
# subjects = ['B','C','D','E']
subjects = ['F','G','H','I']
NUM_SUBJECTS = len(subjects)

runid_names = {
    'B':{
        'runids':["2yhyq225","iy1t5egp"],
        'run_names':["leave B (SVG*)","leave B (Blobs)"]
    },
    'C':{
        'runids':["vten6hg4","1lg477ow"],
        'run_names':["leave C (SVG*)","leave C (Blobs)"]
    },
    'D':{
        'runids':["i91wiw8v","92azggnk"],
        'run_names':["leave D (SVG*)","leave D (Blobs)"]
    },
    'E':{
        'runids':["6bksh0iz","adm76oiv"],
        'run_names':["leave E (SVG*)","leave E (Blobs)"]
    },
    'F':{
        'runids':["c7etn81h","u9e76hxl"],
        'run_names':["leave F (SVG*)","leave F (Blobs)"]
    },
    'G':{
        'runids':["pteyjtdc","72fplaw5"],
        'run_names':["leave G (SVG*)","leave G (Blobs)"]
    },
    'H':{
        'runids':["526eqa92","ay67z0qn"],
        'run_names':["leave H (SVG*)","leave H (Blobs)"]
    },
    'I':{
        'runids':["misunawm","72f7731x"],
        'run_names':["leave I (SVG*)","leave I (Blobs)"]
    }
}

epochs_to_take = 18

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

fig = plt.figure(figsize=(10,6))    
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
                
        plt.subplot(2,2,subject_i+1)
        if i==0:
            plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][0],alpha=1,color='tab:blue')
        else:
            plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][1],alpha=1,color='tab:orange')
        plt.xlabel('Generated Frame Number')
        plt.ylabel('SSIM')
        plt.title(f'Predicted Future-Frame SSIM (leave-{subject}-out)')        
        plt.ylim([0.7,1])
        plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        
        plt.axvline(x=9, color='red',linestyle='--')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2])




# plt.plot(sum_last_epoch_ssim_valid_SVG/NUM_SUBJECTS, marker = 'x', color='red',label='mean')
# plt.plot(sum_last_epoch_ssim_valid_BLOB/NUM_SUBJECTS, marker = 'x', color='blue',label='mean (Blobs)')

plt.tight_layout()
# plt.legend([*[f'leave-{s}-out' for s in subjects],'mean'])

# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[-2:], labels[-2:])

fig.savefig(f'new_SSIM_results_all_position_blob_svg_2.png')
plt.close()



    # fig = plt.figure(figsize=(7,3))
    # for i,runid in enumerate(runids):

    #     _, valid_last_ssim, num_epochs = get_ssim_results_from_wandb(runid)

    #     plt.subplot(1,1,1)
    #     plt.plot(valid_last_ssim.values, label=f'{i+1} (epochs: {num_epochs})')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('SSIM')
    #     plt.title('10th predicted frame SSIM')
    #     plt.ylim([0.7,1])

    # # plt.legend(run_names)
    # plt.tight_layout()
    # fig.savefig(f'new_SSIM_results_during_training_{name}.png')
    # plt.close()




