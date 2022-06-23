import tensorflow
# from inception_resnet_v1_reduction import inference as inception_resnet_v1_reduction
import numpy as np


#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow as v2
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
    # import tensorflow_addons as tfa
    # from tensorflow.keras.layers import Activation
    # from tensorflow.keras.utils import get_custom_objects
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

print_out = True
TCPConnected = False


class PatchEmbedding():
    def __init__(self,
                 image_size: list = [192,192],
                 patch_size: list = [16,16],
                 num_channel: int = 3,
                 embed_length: int = 768,
                 ):
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.embed_length = embed_length
        self.img_conv2D = v2.keras.layers.Conv2D(embed_length,patch_size,strides=patch_size)

    def __call__(self,tf_input):
        net = self.img_conv2D(tf_input)
        net = v2.reshape(net,[-1,self.num_patches,self.embed_length])

        return net

class SelfAttention():
    def __init__(self,hidden_size: int = 768):
        self.key = v2.keras.layers.Dense(hidden_size)
        self.query = v2.keras.layers.Dense(hidden_size)
        self.value = v2.keras.layers.Dense(hidden_size)

    def __call__(self,tensor_input):

        k = self.key(tensor_input)
        q = self.query(tensor_input)
        v = self.value(tensor_input)

        attention_scores = v2.matmul(q,k,transpose_b=True)
        attention_probs = v2.nn.softmax(attention_scores,axis=-1)

        context_layer = v2.matmul(attention_probs, v)

        context_layer = v2.add(context_layer,tensor_input)

        return context_layer

class VitLayer():
    def __init__(self,hidden_size=768,activation=v2.nn.gelu):
        self.layerNorm_1 = v2.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layerNorm_2 = v2.keras.layers.LayerNormalization(epsilon=1e-12)
        self.selfAttention = SelfAttention(hidden_size=hidden_size)
        self.dense = v2.keras.layers.Dense(hidden_size,activation=activation)
        self.hidden_size = hidden_size
    def __call__(self,tensor_input):

        net = self.layerNorm_1(tensor_input)
        net = self.selfAttention(net) + tensor_input

        output_layer = self.layerNorm_2(net)
        output_layer = self.dense(output_layer)

        output_layer += net

        return output_layer

def VitLayer_2(tf_input,hidden_size=768,patch_size=[16,16],drop_rate=0.2):
    patch_num_h = tf_input.shape[1].value // patch_size[0]
    patch_num_w = tf_input.shape[2].value // patch_size[1]
    num_patches = patch_num_h * patch_num_w
    patch_embedding = v2.keras.layers.Conv2D(hidden_size, patch_size, strides=patch_size)(tf_input)
    patch_embedding = v2.reshape(patch_embedding, [-1, num_patches, hidden_size])

    layer = v2.keras.layers.LayerNormalization(epsilon=1e-12)(patch_embedding)
    layer = SelfAttention(hidden_size=hidden_size)(layer)
    layer += patch_embedding

    output_layer = v2.keras.layers.LayerNormalization(epsilon=1e-12)(layer)
    output_layer = v2.keras.layers.Dense(hidden_size * 4, activation=v2.nn.gelu)(output_layer)
    output_layer = v2.keras.layers.Dense(hidden_size, activation=None)(output_layer)
    output_layer = v2.keras.layers.Dropout(drop_rate)(output_layer)
    output_layer += layer

    return v2.reshape(output_layer, [-1, patch_num_h, patch_num_w, hidden_size])
















