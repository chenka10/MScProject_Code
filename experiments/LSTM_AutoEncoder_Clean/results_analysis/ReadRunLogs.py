import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

# runids = ["ij14m0l3","398qrj4w","x6yl07hm", "i9nuriio","bq2onwcn","hetgk0hp", "d4im5xx1","a3di9j14"]


# runids = ["hetgk0hp", "9zijshao" ,"awegarzr","hse2rwhm","nqp7xfd3", "wu8rgstq", "bx6bgjnl", "d5zvd6wc", "wmeaw2hw","i1mrwoir"]
# run_names = ["leave C (SVG)", "leave B", "leave C","leave D","leave E", "leave F", "leave G", "leave H", "leave I","rarp50, leave 37"]

# runids = ["kh4d1ek2", "awegarzr"]
# run_names = ["leave C (SVG)", "leave C"]
# name = "C"

# runids = ["cg7poged", "hse2rwhm"]
# run_names = ["leave D (SVG)", "leave D"]
# name = "D"

# runids = ["lob314nf", "9zijshao"]
# run_names = ["leave B (SVG)", "leave B"]
# name = "B"

runids = ["4ljrmbff", "nqp7xfd3"]
run_names = ["leave E (SVG)", "leave E"]
name = "E"

# runids = ["iss6qjlg", "wu8rgstq"]
# run_names = ["leave F (SVG)", "leave F"]
# name = "F"

# runids = ["4qu8d6zv", "bx6bgjnl"]
# run_names = ["leave G (SVG)", "leave G"]
# name = "G"

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
    plt.ylim([0.5,1])

    plt.subplot(1,2,2)
    plt.plot(last_epoch_ssim_valid.values, marker = 'x', label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Future-Frame Number')
    plt.ylabel('SSIM')
    plt.title('Predicted Future-Frame SSIM')
    plt.ylim([0.5,1])
    plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9],labels=[1,2,3,4,5,6,7,8,9,10])

plt.legend(run_names)
plt.tight_layout()
fig.savefig(f'SSIM_results_{name}.png')
plt.close()

fig = plt.figure(figsize=(7,3))
for i,runid in enumerate(runids):

    _, _, training_last_ssim, valid_last_ssim, num_epochs = get_ssim_results_from_wandb(runid)
    plt.subplot(1,2,1)
    plt.plot(training_last_ssim.values, label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('9th predicted frame SSIM (Train)')
    plt.ylim([0.5,1])

    plt.subplot(1,2,2)
    plt.plot(valid_last_ssim.values, label=f'{i+1} (epochs: {num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('10th predicted frame SSIM')
    plt.ylim([0.5,1])

# plt.legend(run_names)
plt.tight_layout()
fig.savefig(f'SSIM_results_during_training_{name}.png')
plt.close()




