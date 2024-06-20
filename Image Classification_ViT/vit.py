import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.layers import *
from keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super(ClassToken, self).__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value
        )

    def call(self, inputs):


def ViT(cf):

    """Inputs"""
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs= Input(input_shape) ##(None, 256, 3072)
    

    """Patch+ Position Embeddings"""
    patch_embed= Dense(cf['hidden_dim'])(inputs) ##(None, 256, 768)
    # print(patch_embed.shape)

    positions= tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed= Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ##(None, 256, 768)
    # print(pos_embed.shape)

    embed= patch_embed+pos_embed ##(None, 256, 768)


    """Adding Class Token"""

    ClassToken()(embed)

if __name__ == '__main__':
    config = {}
    config['num_layers']= 12
    config['hidden_dim']= 768
    config['num_heads']= 12
    config['mlp_dim']= 3072
    config['num_channels']=3
    config['patch_size']=32
    config['dropout_rate']=0.1
    config['num_patches']=256

    ViT(config)
