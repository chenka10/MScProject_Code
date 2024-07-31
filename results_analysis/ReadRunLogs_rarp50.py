import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

# subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 20
# subjects = ['B','C','D','E']
subjects = ['37']
NUM_SUBJECTS = len(subjects)

# original runs with 10 future frames
# runid_names = {
#     '37':{
#         'runids':["i1mrwoir","12en1cyj"],
#         'run_names':["rarp50 (Blobs)","rarp50 (SVG*)"]
#     }    
# }

# test runs with 20 future frames
runid_names = {
    '37':{
        'runids':["4aoowtb0","izubnmnl"],
        'run_names':["rarp50 (SVG*)", "rarp50 (Blobs)"]
    }    
}

epochs_to_take = 18
epoch_to_examine = -1 # epoch is actually dictated by test runs

def get_ssim_results_from_wandb(runid):
    last_epoch_ssim_train = None
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[epoch_to_examine][[f'valid_SSIM_timestep_{i}' for i in range(NUM_OF_PRED_FRAMES)]]

        valid_last_ssim = hist['valid_SSIM_timestep_9'][:epochs_to_take]

        num_epochs = hist.shape[0]

    return last_epoch_ssim_valid, valid_last_ssim, num_epochs

sum_last_epoch_ssim_valid_BLOB = None
sum_last_epoch_ssim_valid_SVG = None

fig = plt.figure(figsize=(7,4))    
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
                
        # plt.subplot(2,2,subject_i+1)
        if i==0:
            plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][0],alpha=1,color='tab:blue')
        else:
            plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=runid_names[subject]['run_names'][1],alpha=1,color='tab:orange')
        plt.xlabel('Generated Frame Number')
        plt.ylabel('SSIM')
        plt.title(f'Generated Future-Frame SSIM') # (leave-{subject}-out)
        plt.ylim([0.5,0.8])
        plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))                
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2])


plt.tight_layout()


fig.savefig(f'new_SSIM_results_rarp50.png')
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




