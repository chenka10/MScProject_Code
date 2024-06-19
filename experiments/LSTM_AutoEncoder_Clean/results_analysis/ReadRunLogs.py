import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

# runids = ["ij14m0l3","398qrj4w","x6yl07hm", "i9nuriio","bq2onwcn","hetgk0hp", "d4im5xx1","a3di9j14"]
runids = ["hetgk0hp", "a3di9j14","mfeewbl3","awegarzr"]
run_names = ["hetgk0hp", "a3di9j14","mfeewbl3","awegarzr"]
# run_names = ['SVG (w/ position conditioning)', 'Ours']

epochs_to_take = 18

def get_ssim_results_from_wandb(runid):
    last_epoch_ssim_train = None
    last_epoch_ssim_valid = None

    for run in runs:
        if run.id != runid:
            continue

        hist = run.history()    
        last_epoch_ssim_train = hist.iloc[-1][[f'train_SSIM_timestep_{i}' for i in range(10)]]
        last_epoch_ssim_valid = hist.iloc[-3][[f'valid_SSIM_timestep_{i}' for i in range(10)]]

        training_last_ssim = hist['train_SSIM_timestep_9'][:epochs_to_take]
        valid_last_ssim = hist['valid_SSIM_timestep_9'][:epochs_to_take]

        num_epochs = hist.shape[0]

    return last_epoch_ssim_train, last_epoch_ssim_valid, training_last_ssim, valid_last_ssim, num_epochs


fig = plt.figure(figsize=(7,3))
for i,runid in enumerate(runids):

    last_epoch_ssim_train, last_epoch_ssim_valid, _, _, num_epochs = get_ssim_results_from_wandb(runid)
    plt.subplot(1,2,1)
    plt.plot(last_epoch_ssim_train.values, marker = 'x', label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Future-Frame Number')
    plt.ylabel('SSIM')
    plt.title('Predicted Future-Frame SSIM (Train)')
    plt.ylim([0.7,1])

    plt.subplot(1,2,2)
    plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Future-Frame Number')
    plt.ylabel('SSIM')
    plt.title('Predicted Future-Frame SSIM')
    plt.ylim([0.7,1])
    plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9],labels=[1,2,3,4,5,6,7,8,9,10])

plt.legend(run_names)
plt.tight_layout()
fig.savefig(f'SSIM_results.png')
plt.close()

fig = plt.figure(figsize=(7,3))
for i,runid in enumerate(runids):

    _, _, training_last_ssim, valid_last_ssim, num_epochs = get_ssim_results_from_wandb(runid)
    plt.subplot(1,2,1)
    plt.plot(training_last_ssim.values, label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('9th predicted frame SSIM (Train)')
    plt.ylim([0.7,1])

    plt.subplot(1,2,2)
    plt.plot(valid_last_ssim.values, label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('10th predicted frame SSIM')
    plt.ylim([0.7,1])

# plt.legend(run_names)
plt.tight_layout()
fig.savefig(f'SSIM_results_during_training.png')
plt.close()




