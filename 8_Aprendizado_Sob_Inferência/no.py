#=================================================================
#   MYCELIUM.AI V8.0 - NO.PY - APRENDIZADO SOB INFERÊNCIA
#=================================================================

import tensorflow as tf
from tensorflow.keras import layers

class MyceliumNode(layers.Layer):
    def __init__(self, d_model, name=None, **kwargs):
        super(MyceliumNode, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.meta_nerve = layers.Dense(
            d_model, 
            activation='tanh', 
            kernel_initializer='zeros',
            name="meta_nerve"
        )
        self.w1 = layers.Dense(d_model * 4, activation="swish")
        self.w2 = layers.Dense(d_model)

    def call(self, x, training=False, introspection_active=False):
        if introspection_active:
            uncertainty = tf.math.reduce_std(x, axis=-1, keepdims=True)
            meta_signal = self.meta_nerve(uncertainty)
            x = x + meta_signal
            
        res = self.w1(x)
        res = self.w2(res)
        return res

    def get_config(self):
        config = super(MyceliumNode, self).get_config()
        config.update({"d_model": self.d_model})
        return config
