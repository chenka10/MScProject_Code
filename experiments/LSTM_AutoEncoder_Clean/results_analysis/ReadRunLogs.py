import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("chenka/Robotic Surgery MSc")

runid = "ij14m0l3"

last_epoch_ssim_train = None
last_epoch_ssim_valid = None

for run in runs:
    if run.id != runid:
        continue

    hist = run.history()    
    last_epoch_ssim_train = hist.iloc[-1][[f'train_SSIM_timestep_{i}' for i in range(10)]]
    last_epoch_ssim_valid = hist.iloc[-1][[f'valid_SSIM_timestep_{i}' for i in range(10)]]

fig = plt.figure()
plt.plot(last_epoch_ssim_train.values, marker = 'x')
plt.plot(last_epoch_ssim_valid.values, marker = 'x')

plt.xlabel('Future-Frame Number')
plt.ylabel('SSIM')
plt.title('Predicted Future-Frame SSIM')

fig.savefig(f'run_{runid}_SSIM_result.png')
plt.close()




