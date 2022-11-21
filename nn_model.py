from tensorflow.keras.layers import *
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
    self.encoder.add(InputLayer(input_shape = (*nxny_en, 1)))

    for d in range(nol_en):
        if pad == 'symmetric_padding':
            self.encoder.add(symmetric_padding(padding=(1,1)))
        self.encoder.add(Conv2D(2**d*fcv_en, ks_cv_en, **kwargs['encoder']['conv']['kwargs']))
        if bn_en:
            self.encoder.add(BatchNormalization())
        self.encoder.add(Activation(ac_en))
        if p_en == 'average':
            self.encoder.add(AveragePooling2D((2,2)))
        if p_en == 'max':
            self.encoder.add(MaxPooling2D((2,2)))

    ##############################################
    #Decoder
    ##############################################
    self.decoder = tf.keras.Sequential()
    nxny_de = self.encoder.layers[-1].output.shape[1:]
    self.decoder.add(InputLayer(input_shape = nxny_de))
    ##############################################

    for d in range(nol_de)[::-1]:
        if transpose:
            self.decoder.add(Conv2DTranspose(2**d*fcv_de, kernel_size=(2,2), strides=2, padding='same'))
        else:
            if pad == 'symmetric_padding':
                self.decoder.add(symmetric_padding(padding=(1,1)))
            self.decoder.add(Conv2D(2**d*fcv_de, ks_cv_de, **kwargs['decoder']['conv']['kwargs']))
        if bn_de:
            self.decoder.add(BatchNormalization())
        self.decoder.add(Activation(ac_de))
        if up_de:
            self.decoder.add(UpSampling2D((2,2)))

    if pad == 'symmetric_padding':
        self.decoder.add(symmetric_padding(padding=(1,1)))
    self.decoder.add(Conv2D(**kwargs['output_layer']['kwargs']))

  def call(self, input):

    encoded = self.encoder(input)
    decoded = self.decoder(encoded)
    return decoded


def autoencoder_func(**kwargs):

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
    input = Input(shape=(None,None,1))

    for d in range(nol_en):
        if d==0:
            x = input
        if pad == 'symmetric_padding':
            x=symmetric_padding(padding=(1,1))(x)
        x=Conv2D(2**d*fcv_en, ks_cv_en, **kwargs['encoder']['conv']['kwargs'])(x)
        if bn_en:
            x=BatchNormalization()(x)
        x=Activation(ac_en)(x)
        if p_en == 'average':
            x=AveragePooling2D((2,2))(x)
        if p_en == 'max':
            x=MaxPooling2D((2,2))(x)

    ##############################################
    #Decoder
    ##############################################
    for d in range(nol_de)[::-1]:
        if transpose:
            x=Conv2DTranspose(2**d*fcv_de, kernel_size=(2,2), strides=2, padding='same')(x)
        else:
            if pad == 'symmetric_padding':
                x=symmetric_padding(padding=(1,1))(x)
            x=Conv2D(2**d*fcv_de, ks_cv_de, **kwargs['decoder']['conv']['kwargs'])(x)
        if bn_de:
           x= BatchNormalization()(x)
        x=Activation(ac_de)(x)
        if up_de:
           x=UpSampling2D((2,2))(x)

    if pad == 'symmetric_padding':
        x=symmetric_padding(padding=(1,1))(x)
    output = Conv2D(**kwargs['output_layer']['kwargs'])(x)

    model = Model(inputs=[input], outputs=[output])

    return model


class symmetric_padding(Layer):
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(symmetric_padding, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')

    def get_config(self):
        config = super(symmetric_padding, self).get_config()
        config.update({"padding": self.padding})
        return config