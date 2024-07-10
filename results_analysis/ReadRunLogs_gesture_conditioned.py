import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

runid_names = {
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

epochs_to_take = 18

def get_ssim_results_from_wandb(runid):
    last_epoch_ssim_train = None
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[-1][[f'valid_SSIM_timestep_{i}' for i in range(10)]]

        valid_last_ssim = hist['valid_SSIM_timestep_9'][:epochs_to_take]

        num_epochs = hist.shape[0]

    return last_epoch_ssim_valid, valid_last_ssim, num_epochs

sum_last_epoch_ssim_valid_SVG = None
subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 10
# subjects = ['B','C','D','E']
NUM_SUBJECTS = len(subjects)
fig = plt.figure(figsize=(8,4))    
for subject_i,subject in enumerate(subjects):

    runids = runid_names[subject]['runids']
    run_names = runid_names[subject]['run_names']
    name = subject    

    for i,runid in enumerate(runids):

        last_epoch_ssim_valid, _, num_epochs = get_ssim_results_from_wandb(runid)  

        if sum_last_epoch_ssim_valid_SVG is None:            
            sum_last_epoch_ssim_valid_SVG = last_epoch_ssim_valid.values            
        else:       
            sum_last_epoch_ssim_valid_SVG += last_epoch_ssim_valid.values
                
        # plt.subplot(2,2,subject_i+1)
        plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=f'{i+1} (epochs: {num_epochs})',alpha=0.2)
        plt.xlabel('Generated Frame Number')
        plt.ylabel('SSIM')
        plt.title(f'Predicted Future-Frame SSIM (leave-{subject}-out)')        
        plt.ylim([0.7,1])
        plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        




plt.plot(sum_last_epoch_ssim_valid_SVG/NUM_SUBJECTS, marker = 'x', label=f'{i+1} (epochs: {num_epochs})',color='blue')
plt.tight_layout()
plt.legend([*[f'leave-{s}-out' for s in subjects],'mean'])
fig.savefig(f'new_SSIM_results_all_gestures_first.png')
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




