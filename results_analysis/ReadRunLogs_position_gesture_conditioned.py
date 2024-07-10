import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

runid_names_gesture = {
    'B':{
        'runids':["w5dwxel6"],
        'run_names':["leave B"]
    },
    'C':{
        'runids':["0jrghc8b"],
        'run_names':["leave C"]
    },
    'D':{
        'runids':["bk6eh1ad"],
        'run_names':["leave D"]
    },
    'E':{
        'runids':["i50c21wg"],
        'run_names':["leave E"]
    },
    'F':{
        'runids':["c5k8s5b3"],
        'run_names':["leave F"]
    },
    'G':{
        'runids':["ywhveyj4"],
        'run_names':["leave G"]
    },
    'H':{
        'runids':["hnxkwxm2"],
        'run_names':["leave H"]
    },
    'I':{
        'runids':["gsb1fpmx"],
        'run_names':["leave I"]
    }
}

runid_names_position = {
    'B':{
        'runids':["lob314nf"],
        'run_names':["leave B"]
    },
    'C':{
        'runids':["kh4d1ek2"],
        'run_names':["leave C"]
    },
    'D':{
        'runids':["cg7poged"],
        'run_names':["leave D"]
    },
    'E':{
        'runids':["4ljrmbff"],
        'run_names':["leave E"]
    },
    'F':{
        'runids':["iss6qjlg"],
        'run_names':["leave F"]
    },
    'G':{
        'runids':["4qu8d6zv"],
        'run_names':["leave G"]
    },
    'H':{
        'runids':["ihjq13i1"],
        'run_names':["leave H"]
    },
    'I':{
        'runids':["l7yeh4hd"],
        'run_names':["leave I"]
    }
}

epochs_to_take = 18

def get_ssim_results_from_wandb(runid):    
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[-1][[f'valid_SSIM_timestep_{i}' for i in range(10)]]

        valid_last_ssim = hist['valid_SSIM_timestep_9'][:epochs_to_take]

        num_epochs = hist.shape[0]

    return last_epoch_ssim_valid, valid_last_ssim, num_epochs

fig = plt.figure(figsize=(8,4))    

for iter in [0,1]:
    sum_last_epoch_ssim_valid_SVG = None
    subjects = ['B','C','D','E','F','G','H','I']
    NUM_OF_PRED_FRAMES = 10
    # subjects = ['B','C','D','E']
    NUM_SUBJECTS = len(subjects)    
    for subject_i,subject in enumerate(subjects):

        if iter==0:
            runids = runid_names_position[subject]['runids']
            run_names = runid_names_position[subject]['run_names']
        else:
            runids = runid_names_gesture[subject]['runids']
            run_names = runid_names_gesture[subject]['run_names']
        name = subject    

        for i,runid in enumerate(runids):

            last_epoch_ssim_valid, _, num_epochs = get_ssim_results_from_wandb(runid)  

            if sum_last_epoch_ssim_valid_SVG is None:            
                sum_last_epoch_ssim_valid_SVG = last_epoch_ssim_valid.values            
            else:       
                sum_last_epoch_ssim_valid_SVG += last_epoch_ssim_valid.values
                    
            # plt.subplot(2,2,subject_i+1)
            # plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=f'{i+1} (epochs: {num_epochs})',alpha=0.2)
            # plt.xlabel('Generated Frame Number')
            # plt.ylabel('SSIM')
            # plt.title(f'Predicted Future-Frame SSIM (leave-{subject}-out)')        
            # plt.ylim([0.7,1])
            # plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        


    plt.plot(sum_last_epoch_ssim_valid_SVG/NUM_SUBJECTS, marker = 'x', label=f'{i+1} (epochs: {num_epochs})')


plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1))) 
plt.ylim([0.7,1]) 
plt.xlabel('Generated Frame Number')
plt.ylabel('SSIM')      
plt.legend(['mean (position)','mean (gesture)'])
plt.title(f'Predicted Future-Frame SSIM Mean for Position/Gesture Conditioning')        
plt.tight_layout()
fig.savefig(f'new_SSIM_results_all_position_condition_mean.png')
plt.close()