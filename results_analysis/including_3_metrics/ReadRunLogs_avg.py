import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

metrics = ['SSIM','PSNR','LPIPS']
subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 20
NUM_SUBJECTS = len(subjects)
runid_names = {
    'B':{
        'runids':["uj8pfyjb","cnp06uk0"],
        'run_names':["SVG*","Blobs"]
    },
    'C':{
        'runids':["3it7q5ak","skong5ml"],
        'run_names':["SVG*","Blobs"]
    },
    'D':{
        'runids':["rekkgh78","7u0ghl9x"],
        'run_names':["SVG*","Blobs"]
    },
    'E':{
        'runids':["zrn1pmrq","79os7b2h"],
        'run_names':["SVG*","Blobs"]
    },
    'F':{
        'runids':["ydjrpmjo","1aesxdcz"],
        'run_names':["SVG*","Blobs"]
    },
    'G':{
        'runids':["4ylkoh7h","16djez5b"],
        'run_names':["SVG*","Blobs"]
    },
    'H':{
        'runids':["5cnz5fw8","wg2tfbll"],
        'run_names':["SVG*","Blobs"]
    },
    'I':{
        'runids':["1i9sogae","vqjic0v5"],
        'run_names':["SVG*","Blobs"]
    }
}
epochs_to_take = 18

def get_metric_results_from_wandb(runid,metric):
    last_epoch_ssim_train = None
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[-1][[f'valid_{metric}_timestep_{i}' for i in range(NUM_OF_PRED_FRAMES)]]

    return last_epoch_ssim_valid

fig = plt.figure(figsize=(14,3))  
i_plot = 0

for metric in metrics:

    sum_last_epoch_ssim_valid_BLOB = None
    sum_last_epoch_ssim_valid_SVG = None

    i_plot+=1
    plt.subplot(1,3,i_plot)
    for subject_i,subject in enumerate(subjects):

        runids = runid_names[subject]['runids']
        run_names = runid_names[subject]['run_names']
        name = subject    

        for i,runid in enumerate(runids):

            last_epoch_metric_valid = get_metric_results_from_wandb(runid, metric)  

            if sum_last_epoch_ssim_valid_BLOB is None:    
                if i==0:        
                    sum_last_epoch_ssim_valid_SVG = last_epoch_metric_valid.values            
                else:
                    sum_last_epoch_ssim_valid_BLOB = last_epoch_metric_valid.values            
            else:       
                if i==0:        
                    sum_last_epoch_ssim_valid_SVG += last_epoch_metric_valid.values            
                else:
                    sum_last_epoch_ssim_valid_BLOB += last_epoch_metric_valid.values 
    
    plt.xlabel('Generated Frame Number')
    plt.ylabel(f'{metric}')        
    # plt.ylim([0.7,1])
    plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        

    plt.plot(sum_last_epoch_ssim_valid_SVG/NUM_SUBJECTS, marker = 'x', color='tab:blue',label='mean (SVG*)')
    plt.plot(sum_last_epoch_ssim_valid_BLOB/NUM_SUBJECTS, marker = 'x', color='tab:orange',label='mean (Blobs)')
    plt.axvline(x=9,color='red',linestyle='--',label='last pred. during training')   
    plt.legend()



plt.suptitle(f'Generated Future-Frame Comparison (LOUO folds mean)')  
plt.tight_layout()
fig.savefig(f'new_results_all_position_blob_svg_avg.png')
plt.close()


