#=================================================================
#   MYCELIUM.AI V8.0 - CAMINHO.PY - APRENDIZADO SOB INFERÊNCIA
#=================================================================

import tensorflow as tf
from tensorflow.keras import layers

class MyceliumPath(layers.Layer):
    def __init__(self, d_model, name=None, **kwargs):
        super(MyceliumPath, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.w_stone = self.add_weight(
            shape=(dim, self.d_model), 
            initializer="orthogonal", 
            trainable=True, 
            name="stone"
        )
        self.w_sand = self.add_weight(
            shape=(dim, self.d_model), 
            initializer="zeros", 
            trainable=False, 
            name="sand"
        )
        self.gate = self.add_weight(
            shape=(self.d_model,), 
            initializer="ones", 
            trainable=True, 
            name="gate"
        )
        self.norm.build(input_shape)
        super(MyceliumPath, self).build(input_shape)

    def call(self, x, external_state=None):
        x_norm = self.norm(x)
        w_q = tf.stop_gradient(tf.round(self.w_stone * 127.0) / 127.0 - self.w_stone) + self.w_stone
        current_sand = external_state if external_state is not None else self.w_sand
        sand_clamped = tf.clip_by_value(current_sand, -0.3, 0.3)
        res = tf.matmul(x_norm, w_q + sand_clamped) / (self.d_model ** 0.5)
        
        delta = tf.matmul(tf.transpose(x_norm, perm=[0, 2, 1]), x_norm)
        seq_len = tf.cast(tf.shape(x)[1], tf.float32)
        new_sand = (current_sand * 0.99) + (0.01 * delta / seq_len)
        new_sand = tf.clip_by_value(new_sand, -0.5, 0.5)

        return (res * tf.nn.sigmoid(self.gate)), new_sand

    def get_config(self):
        config = super(MyceliumPath, self).get_config()
        config.update({"d_model": self.d_model})
        return config