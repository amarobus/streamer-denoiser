from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import pad
import tensorflow as tf

class autoencoder(Model):
  def __init__(self, **kwargs):
    super(autoencoder, self).__init__()

    pad = kwargs['custom_padding']
    
    nxny_en = kwargs['input_shape']
    fcv_en = kwargs['encoder']['conv']['filters']
    ks_cv_en = kwargs['encoder']['conv']['kernel_size']
    nol_en = kwargs['encoder']['num_layers']
    ac_en = kwargs['encoder']['activation']
    p_en= kwargs['encoder']['pooling']
    bn_en = kwargs['encoder']['batch_normalization']

    fcv_de = kwargs['decoder']['conv']['filters']
    ks_cv_de = kwargs['decoder']['conv']['kernel_size']
    nol_de = kwargs['decoder']['num_layers']
    ac_de = kwargs['decoder']['activation']
    up_de = kwargs['decoder']['upsampling']
    bn_de = kwargs['decoder']['batch_normalization']
    transpose = kwargs['decoder']['transpose']

    ##############################################
    #Encoder
    ##############################################
    self.encoder = tf.keras.Sequential()
    self.encoder.add(layers.InputLayer(input_shape = (*nxny_en, 1)))

    for d in range(nol_en):
        if pad == 'symmetric_padding':
            self.encoder.add(symmetric_padding(padding=(1,1)))
        self.encoder.add(layers.Conv2D(2**d*fcv_en, ks_cv_en, **kwargs['encoder']['conv']['kwargs']))
        if bn_en:
            self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Activation(ac_en))
        if p_en == 'average':
            self.encoder.add(layers.AveragePooling2D((2,2)))
        if p_en == 'max':
            self.encoder.add(layers.MaxPooling2D((2,2)))

    ##############################################
    #Decoder
    ##############################################
    self.decoder = tf.keras.Sequential()
    nxny_de = self.encoder.layers[-1].output.shape[1:]
    self.decoder.add(layers.InputLayer(input_shape = nxny_de))
    ##############################################

    for d in range(nol_de)[::-1]:
        if transpose:
            self.decoder.add(layers.Conv2DTranspose(2**d*fcv_de, kernel_size=(2,2), strides=2, padding='same'))
        else:
            if pad == 'symmetric_padding':
                self.decoder.add(symmetric_padding(padding=(1,1)))
            self.decoder.add(layers.Conv2D(2**d*fcv_de, ks_cv_de, **kwargs['decoder']['conv']['kwargs']))
        if bn_de:
            self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.Activation(ac_de))
        if up_de:
            self.decoder.add(layers.UpSampling2D((2,2)))

    if pad == 'symmetric_padding':
        self.decoder.add(symmetric_padding(padding=(1,1)))
    self.decoder.add(layers.Conv2D(**kwargs['output_layer']['kwargs']))

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
