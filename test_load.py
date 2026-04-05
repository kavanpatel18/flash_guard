import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import layers
import tensorflow as tf

class TemporalAttention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)

    def call(self, hidden_states, training=False):
        score   = self.v(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg

try:
    m = keras.models.load_model('msa_gru_best.keras', compile=False, custom_objects={'TemporalAttention': TemporalAttention})
    print("SUCCESS")
    print(m.input_shape)
except Exception as e:
    import traceback
    traceback.print_exc()
