#=================================================================
#   MYCELIUM.AI V7.0 - NO.PY - PROCESSAMENTO PARALELO
#=================================================================


import tensorflow as tf
from tensorflow.keras import layers

class MyceliumNode(layers.Layer):
    """
    Nó de processamento da MyceliumAI.
    Encapsula: Transformação -> Normalização -> Ativação.
    """
    def __init__(self, layer_type, filters, kernel_size=None, strides=(1,1), activation="relu", use_bn=True, **kwargs):
        super(MyceliumNode, self).__init__(**kwargs)
        self.use_bn = use_bn
        
        # Define o tipo de processamento do núcleo
        if layer_type == "Dense":
            self.core = layers.Dense(filters, use_bias=not use_bn)
        elif layer_type == "Conv2DTranspose":
            self.core = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=not use_bn)
        elif layer_type == "Conv2D":
            self.core = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=not use_bn)
            
        # Componentes auxiliares
        if use_bn:
            self.bn = layers.BatchNormalization()
        
        if activation == "leaky_relu":
            self.act = layers.LeakyReLU(alpha=0.2)
        else:
            self.act = layers.Activation(activation)

    def call(self, inputs, training=False):
        x = self.core(inputs)
        if self.use_bn:
            x = self.bn(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super(MyceliumNode, self).get_config()
        # Adicionar parâmetros ao config se precisar salvar/carregar
        return config