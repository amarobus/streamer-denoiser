import glob
import time
import random
import numpy as np
import tensorflow as tf

import data_generator
import nn_model
import utils

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

def main():

    # Model settings
    input_shape = (4096, 768)
    layers = 3
    loss_name = 'MSE'
    model_name = 'autoencoder'
    batch_normalization = False
    max_pooling = True
    average_pooling = False
    upsampling = True
    transpose = False
    filters = 16
    kernel_size = 3
    activation = 'relu'
    batch_size = 12

    kwargs = {'input_shape': (*input_shape,1),
        'filters': filters,
        'kernel_size': kernel_size,
        'activation': activation,
        'encoder_kwargs' : {"padding": "valid", "strides": 1},
        'decoder_kwargs' : {"padding": "valid", "strides": 1},
        'num_layers': layers,
        'output_kwargs': {"padding": "valid", "strides": 1, "activation": "tanh"},
        'batch_normalization': batch_normalization,
        'max_pooling': max_pooling,
        'average_pooling': average_pooling,
        'upsampling': upsampling,
        'transpose': transpose}


    # Generate output name and title
    output_name = f'{model_name}_{filters}F{kernel_size}'
    title = f'Model name: {model_name}\n{filters} Filters {kernel_size}x{kernel_size}'

    title += f'\nLayers: {layers} \nLoss: {loss_name}'
    output_name += f'_L{layers}_{loss_name}'

    if max_pooling:
        output_name += '_MP'
        title += f'\nDownsampling: MaxPooling'
    if average_pooling:
        output_name += '_AP'
        title += f'\nDownsampling: AveragePooling'
    if batch_normalization:
        output_name += '_BN'
        title += f'\nBatch Normalization'
    if upsampling:
        output_name += '_US'
        title += f'\nUpsampling'
    if transpose:
        output_name += '_T'
        title += f'\nTranspose'

    with open(f'{output_name}.yaml', 'w') as file:
        yaml.dump(kwargs, file)

    # Training, validation, and test data generators
    training_generator = data_generator.DataGenerator(batch_size = batch_size, shuffle = True)
    validation_generator = data_generator.DataGenerator(batch_size = batch_size, shuffle = True, valid = True)
    test_generator = data_generator.DataGenerator(batch_size = batch_size, shuffle=True, test = True)

    # Training
    model = nn_model.autoencoder(**kwargs)
    loss = tf.keras.losses.MeanSquaredError()
    model = training(model, loss, output_name, training_generator, validation_generator)

    try:
        model = tf.keras.models.load_model(f'./checkpoints/{output_name}')
    except:
        pass

    # Evaluation
    test_loss = model.evaluate(test_generator)
    print(f'Test loss: {test_loss}')

    # Plots to visualize results
    X_test, Y_test = test_generator.__getitem__(0)
    utils.plot_original_clean(model, X_test, Y_test, title, output_name)
    utils.plot_history(model, title, output_name)


def training(model, loss, output_name, training_generator, validation_generator):

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0025)
    model.compile(optimizer=opt, loss=loss)

    # Callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(f'./checkpoints/{output_name}',
                                                            monitor='val_loss', mode='min',save_best_only=True)

    start = time.time()
    history = model.fit(training_generator,
                    epochs=1000,
                    #shuffle=False,
                    validation_data=validation_generator,
                    callbacks=[es_callback, reduce_lr_callback, checkpoint_callback, utils.plot_losses()],
                    use_multiprocessing = True,
                    workers = 6)
    end = time.time()
    print(f'Training: {(end - start)/60.} min')

    model.encoder.summary()
    model.decoder.summary()


    np.save(f'./checkpoints/{output_name}/history.npy',history.history)


    # Generate file with model performance
    min_loss = np.amin(history.history['loss'])
    min_val_loss = np.amin(history.history['val_loss'])

    output_text = f'Model name\tTraining time (min)\tLoss\tValidation loss\n{output_name}\t{(end-start)/60.}\t{min_loss}\t{min_val_loss}'

    f = open('model_performance.txt','a')
    f.write(output_text)
    f.close()

    return model


if __name__ == '__main__':
    main()
