from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import pad
import tensorflow as tf

class autoencoder(Model):
  def __init__(self, **kwargs):
    super(autoencoder, self).__init__()
    filters = kwargs['filters']
    kernel_size = kwargs['kernel_size']
    num_layers = kwargs['num_layers']
    encoder_input_shape = kwargs['input_shape']
    average_pooling = kwargs['average_pooling']
    max_pooling = kwargs['max_pooling']
    upsampling = kwargs['upsampling']
    batch_normalization = kwargs['batch_normalization']
    transpose = kwargs['transpose']
    activation = kwargs['activation']

    ##############################################
    #Encoder
    ##############################################
    self.encoder = tf.keras.Sequential()
    self.encoder.add(layers.InputLayer(input_shape = encoder_input_shape))

    for d in range(num_layers):
        self.encoder.add(symmetric_padding(padding=(1,1)))
        self.encoder.add(layers.Conv2D(2**d*filters, kernel_size, **kwargs['encoder_kwargs']))
        if batch_normalization:
            self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Activation('relu'))
        if average_pooling:
            self.encoder.add(layers.AveragePooling2D((2,2)))
        if max_pooling:
            self.encoder.add(layers.MaxPooling2D((2,2)))

    ##############################################
    #Decoder
    ##############################################
    self.decoder = tf.keras.Sequential()
    decoder_input_shape = self.encoder.layers[-1].output.shape[1:]
    self.decoder.add(layers.InputLayer(input_shape = decoder_input_shape))
    ##############################################

    for d in range(num_layers)[::-1]:
        if transpose:
            self.decoder.add(layers.Conv2DTranspose(2**d*filters, kernel_size=(2,2), strides=2, padding='same'))
        else:
            self.decoder.add(sym_padding(padding=(1,1)))
            self.decoder.add(layers.Conv2D(2**d*filters, kernel_size, **kwargs['decoder_kwargs']))
        if batch_normalization:
            self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.Activation('relu'))
        if upsampling:
            self.decoder.add(layers.UpSampling2D((2,2)))


    self.decoder.add(sym_padding(padding=(1,1)))
    self.decoder.add(layers.Conv2D(1, 3, **kwargs['output_kwargs']))

  def call(self, input):

    encoded = self.encoder(input)
    decoded = self.decoder(encoded)
    return decoded




class symmetric_padding(layers.Layer):
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(symmetric_padding, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')
