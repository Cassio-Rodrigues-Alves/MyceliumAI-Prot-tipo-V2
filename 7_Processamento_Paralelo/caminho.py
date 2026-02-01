#=================================================================
#   MYCELIUM.AI V7.0 - CAMINHO.PY - PROCESSAMENTO PARALELO
#=================================================================

import tensorflow as tf
from tensorflow.keras import layers

class MyceliumPath(layers.Layer):
    """
    Implementação da arquitetura Mycelium: y = x * w + b (Channel-wise).
    Realiza modulação seletiva de canais antes do processamento do nó.
    """
    def __init__(self, name="caminho", **kwargs):
        super(MyceliumPath, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # Aprende um peso e bias para cada canal de feature individualmente
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        # Multiplicação element-wise (broadcasting) + soma
        return (inputs * self.w) + self.b

    def get_config(self):
        config = super(MyceliumPath, self).get_config()
        return config