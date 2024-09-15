import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

# subjects = ['B','C','D','E','F','G','H','I']
NUM_OF_PRED_FRAMES = 20
metrics = ['SSIM','PSNR','LPIPS']
# subjects = ['B','C','D','E']
subjects = ['F','G','H','I']
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

def get_metric_results_from_wandb(runid, metric):    
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_valid = hist.iloc[-1][[f'valid_{metric}_timestep_{i}' for i in range(NUM_OF_PRED_FRAMES)]]

    return last_epoch_ssim_valid

for metric in metrics:
    fig = plt.figure(figsize=(10,6))    
    
    sum_last_epoch_ssim_valid_BLOB = None
    sum_last_epoch_ssim_valid_SVG = None

    for subject_i,subject in enumerate(subjects):

        runids = runid_names[subject]['runids']
        run_names = runid_names[subject]['run_names']
        name = subject    

        for i,runid in enumerate(runids):

            last_epoch_ssim_valid = get_metric_results_from_wandb(runid, metric)  

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
            plt.ylabel(f'{metric}')
            plt.title(f'Predicted Future-Frame {metric} (leave-{subject}-out)')        
            # plt.ylim([0.7,1])
            plt.xticks(ticks = list(range(NUM_OF_PRED_FRAMES)),labels=list(range(1,NUM_OF_PRED_FRAMES + 1)))        
            plt.axvline(x=9, color='red',linestyle='--')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:2], labels[:2])

    plt.tight_layout()

    fig.savefig(f'new_{metric}_results_all_position_blob_svg_3.png')
    plt.close()





