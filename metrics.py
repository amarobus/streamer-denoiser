import tensorflow as tf
import numpy as np


class MSEC(tf.keras.metrics.Metric):
    def __init__(self, r, dr, dz, name='msec', **kwargs):
        super(MSEC, self).__init__(name=name, **kwargs)
        self.msec = self.add_weight(name='msec', initializer='zeros')
        self.counter =  self.add_weight(name='counter', initializer='zeros')
        self.r = r
        self.dr = dr
        self.dz = dz


    def update_state(self, y_true, y_pred, sample_weight=None):

        charge_pred = 2 * np.pi * tf.reduce_sum(tf.multiply(y_pred, self.r), axis=[1,2]) * self.dr * self.dz
        charge_true = 2 * np.pi * tf.reduce_sum(tf.multiply(y_true, self.r), axis=[1,2]) * self.dr * self.dz
      
        squared_difference = tf.square(charge_pred - charge_true)

        self.counter.assign_add(1)
        self.msec.assign_add(tf.reduce_mean(squared_difference))
        

    def result(self):
        return self.msec/self.counter


    def reset_state(self):
        self.msec.assign(0)
        self.counter.assign(0)


def wrapper_msec(r, dr, dz):
    def msec(y_true,y_pred):
        charge_pred = 2 * np.pi * tf.reduce_sum(tf.multiply(y_pred, r), axis=[1,2]) * dr * dz
        charge_true = 2 * np.pi * tf.reduce_sum(tf.multiply(y_true, r), axis=[1,2]) * dr * dz

        squared_difference = tf.square(charge_pred - charge_true)

        return tf.reduce_mean(squared_difference)
    return msec
