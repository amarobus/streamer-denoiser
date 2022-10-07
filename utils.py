import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

def plot_original_clean(model, X, Y, output_name):

    try:
        os.mkdir('images')
    except:
        pass

    print('Plotting results for a few test samples...')
    for index in range(len(X)):
        start = time.time()
        prediction = model.predict(X[index][None,...])
        end = time.time()
        print(f'Inference time: {end - start} s')

        # Plot result
        comparison = np.append(np.flip(prediction.squeeze(), axis=1), Y[index].squeeze(), axis=1)
        max_abs = np.amax(np.abs(comparison))
        plt.matshow(comparison.T, cmap='seismic', interpolation='none', vmin = -max_abs, vmax = max_abs)
        plt.title(f'Denoised range: [{np.amin(prediction)},{np.amax(prediction)}]\n Original range: [{np.amin(Y[index])},{np.amax(Y[index])}]')
        plt.annotate('Clean', (0.1,0.7), xycoords='figure fraction', size=20, color='White')
        plt.annotate('Original', (0.1,0.2), xycoords='figure fraction', size=20, color='White')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f'images/{index}_{output_name}.png', dpi=200, bbox_inches='tight')
        plt.close()


def plot_errors(model, X, Y, output_name):

    try:
        os.mkdir('erros')
    except:
        pass

    print('Plotting errors for a few test samples...')
    for index in range(len(X)):
        start = time.time()
        prediction = model.predict(X[index][None,...])
        end = time.time()
        print(f'Inference time: {end - start} s')
        error = (prediction.squeeze() - Y[index].squeeze())**2

        # Plot result
        comparison = np.append(np.flip(error, axis=1), error, axis=1)
        plt.matshow(comparison.T, cmap='hot', interpolation='none')
        plt.title(f'Denoised range: [{np.amin(prediction)},{np.amax(prediction)}]\n Original range: [{np.amin(Y[index])},{np.amax(Y[index])}]')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f'errors/{index}_{output_name}.png', dpi=200, bbox_inches='tight')
        plt.close()


def plot_history(model, output_name):
    try:
        os.mkdir('losses')
    except:
        pass
    history=np.load(f'./checkpoints/{output_name}/history.npy',allow_pickle='TRUE').item()
    pd.DataFrame(history).plot()
    plt.grid()
    plt.savefig(f'losses/{output_name}.png', dpi=200, bbox_inches='tight')
    plt.close()


class plot_losses(tf.keras.callbacks.Callback):

    def __init__(self):
        self.training_loss = []
        self.valid_loss = []

    def on_epoch_end(self, epoch, logs = None):
        self.training_loss.append(logs['loss'])
        self.valid_loss.append(logs['val_loss'])

        it = np.arange(1, len(self.training_loss) + 1, 1)

        plt.plot(it, self.training_loss, label='Training')
        plt.plot(it, self.valid_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Losses.png', dpi=200)
        plt.close()
