import glob
import time
import random
import numpy as np
import tensorflow as tf
import yaml
import pandas as pd
import argparse

import data_generator
import nn_model
import utils

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yaml", required=True,
help="path to the model settings file")
args = vars(ap.parse_args())

def main():

    # Mask to be applied to the loaded data
    mask = (2304, None, None, None)

    # Model settings
    with open(args['yaml']) as file: 
        # The FullLoader parameter handles the conversion from YAML 
        # scalar values to Python the dictionary format 
        kwargs = yaml.load(file, Loader=yaml.FullLoader) 

    output_name = kwargs['output_name']
    batch_size = kwargs['batch_size']
    lr = kwargs['learning_rate']
    epochs = kwargs['epochs']
    input_shape = kwargs['input_shape']

    # Write settings in a file
    with open(f'{output_name}.yaml', 'w') as file:
        yaml.dump(kwargs, file)


    # Training, validation, and test data generators
    training_generator = data_generator.DataGenerator(paths = kwargs['paths'], batch_size = batch_size, 
                                                        
                                                        dim = input_shape, mask = mask, shuffle = True)

    validation_generator = data_generator.DataGenerator(paths = kwargs['paths'], batch_size = batch_size, 
                                                        
                                                        dim = input_shape, mask = mask,  shuffle = True, valid = True)

    test_generator = data_generator.DataGenerator(paths = kwargs['paths'], batch_size = batch_size, 
                                                        
                                                        dim = input_shape, mask = mask, shuffle=True, test = True)

    # Initiate model structure
    model = nn_model.autoencoder(**kwargs)

    # Compile model
    if kwargs['loss'] == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif kwargs['loss'] == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=opt, loss=loss)

    # Callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, verbose=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(f'./checkpoints/{output_name}', verbose=1,
                                                            monitor='val_loss', mode='min',save_best_only=True)

    # Training
    start = time.time()
    history = model.fit(training_generator,
                    epochs=epochs,
                    #shuffle=False,
                    validation_data=validation_generator,
                    callbacks=[es_callback, reduce_lr_callback, checkpoint_callback, utils.plot_losses()],
                    use_multiprocessing = True,
                    workers = 6)
    end = time.time()
    print(f'Training: {(end - start)/60.} min')

    # Save history
    np.save(f'./checkpoints/{output_name}/history.npy',history.history)
    min_loss = np.amin(history.history['loss'])
    min_val_loss = np.amin(history.history['val_loss'])

    # Dataframe with model stats
    model_stats = pd.DataFrame({'Model':[output_name], 'Time':(end-start)/60., 'Loss':min_loss, 'Valid Loss':min_val_loss})

    # Load last checkpoint
    try:
        model = tf.keras.models.load_model(f'./checkpoints/{output_name}')
        print('Load model (best checkpoint): Successful')
    except:
        pass

    # Evaluation
    print('Evaluating model on the test dataset...')
    test_loss = model.evaluate(test_generator)
    print(f'Test loss: {test_loss}')

    model_stats['Test Loss'] = test_loss

    # Write file with model performance
    print('Saving model performance...')
    model_stats.to_csv('model_performance.txt', index=None, header=None, sep=' ', mode='a')

    # Plots to visualize results
    X_test, Y_test = test_generator.__getitem__(0)
    utils.plot_original_clean(model, X_test, Y_test, output_name)
    utils.plot_errors(model, X_test, Y_test, output_name)
    utils.plot_history(model, output_name)


if __name__ == '__main__':
    main()
