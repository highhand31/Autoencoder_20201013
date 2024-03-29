import tensorflow,math
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
                 kernel: tuple = (3,3),
                 stride: tuple = (2,2),
                 embed_len: int = 32,
                 padding: str = 'same'
                 ):
        # self.num_patches = (image_size[0] // stride[0]) * (image_size[1] // stride[1])
        self.stride = stride
        self.embed_len = embed_len
        self.conv2D = tf.keras.layers.Conv2D(embed_len,kernel,strides=stride,padding=padding)

    def __call__(self,x):
        net = self.conv2D(x)
        H = x.shape[1].value // self.stride[0]
        W = x.shape[2].value // self.stride[1]
        out_size = (H,W)
        num_patch = H * W
        net = tf.reshape(net,[-1,num_patch,self.embed_len])
        print("PatchEmbedding shape:",net.shape)
        print("out_size:",out_size)

        return net,out_size

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # print("q shape:", q.shape)
  # print("k shape:", k.shape)
  # print("v shape:", v.shape)
  # print("matmul_qk shape:", matmul_qk.shape)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  # dk = tf.cast(k.shape[-1].value, tf.float32)
  sqrt = tf.math.sqrt(dk)
  scaled_attention_logits = matmul_qk / sqrt
  # scaled_attention_logits = matmul_qk / 5.2

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def nlc_to_nhwc(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3, 'input shape dim != 3'
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    # return x.transpose(1, 2).reshape(B, C, H, W)
    return tf.reshape(x,[-1,H,W,C])

def nhwc_to_nlc(x):
    """Flatten [N, H, W, C] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, H, W, C] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4, 'input shape dim != 4'
    L = x.shape[1].value * x.shape[2].value
    C = x.shape[-1].value

    # return x.flatten(2).transpose(1, 2).contiguous()
    return tf.reshape(x,[-1,L,C])

class MultiheadAttention():
    def __init__(self,hidden_size: int = 768,num_heads: int = 1):
        assert hidden_size % num_heads == 0, "hidden_size % num_head != 0"

        self.depth = hidden_size // num_heads
        self.key = tf.keras.layers.Dense(hidden_size)
        self.query = tf.keras.layers.Dense(hidden_size)
        self.value = tf.keras.layers.Dense(hidden_size)
        self.dense = tf.keras.layers.Dense(hidden_size)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.sqrt_att_head_size = math.sqrt(num_heads)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, hidden_size_t, depth)
        """
        x = tf.reshape(x,[-1,self.num_heads,x.shape[1].value//self.depth,self.depth])

        return x
    def transpose_for_scores(self, x) :
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        seq_length = x.shape[1].value
        #print("seq_length:",seq_length)
        tensor = tf.reshape(tensor=x, shape=(-1, seq_length, self.num_heads, self.depth))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def __call__(self,tensor_q,tensor_k,tensor_v,mask=None):
        # batch_size = tensor_q.shape[0].value
        num_patch = tensor_q.shape[1].value
        q = self.query(tensor_q)  # (batch_size, num_patch, hidden_size)
        k = self.key(tensor_k) #(batch_size, num_patch, hidden_size)
        v = self.value(tensor_v) #(batch_size, num_patch, hidden_size)
        print("(batch_size, num_patch, hidden_size)")
        print("q shape:", q.shape)
        print("k shape:", k.shape)
        print("v shape:", v.shape)


        # q = self.split_heads(q)  # (batch_size, num_heads, hidden_size_q, depth)
        # k = self.split_heads(k)  # (batch_size, num_heads, hidden_size_k, depth)
        # v = self.split_heads(v)  # (batch_size, num_heads, hidden_size_v, depth)
        q = self.transpose_for_scores(q)  # (batch_size, num_heads, hidden_size_q, depth)
        k = self.transpose_for_scores(k)  # (batch_size, num_heads, hidden_size_k, depth)
        v = self.transpose_for_scores(v)  # (batch_size, num_heads, hidden_size_v, depth)
        print("(batch_size, num_heads, hidden_size_q, depth)")
        print("q shape:", q.shape)
        print("k shape:", k.shape)
        print("v shape:", v.shape)

        # scaled_attention.shape == (batch_size, num_heads, num_patch, depth)
        # attention_weights.shape == (batch_size, num_heads, num_patch, seq_len_k)
        # scaled_attention, attention_weights = scaled_dot_product_attention(
        #     q, k, v, mask)
        attention_scores = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)#(batch_size, num_heads, num_patch, seq_len_k)
        print("after q k dot product:", attention_probs.shape)
        # Mask heads if we want to
        if mask is not None:
            attention_probs = tf.multiply(attention_probs, mask)

        attention_output = tf.matmul(attention_probs, v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_output,
                                      (-1, num_patch, self.hidden_size))  # (batch_size, num_patch, hidden_size)

        output = self.dense(concat_attention)  # (batch_size, num_patch, hidden_size)
        print("(batch_size, num_patch, hidden_size)")
        print("attention_output shape:",output.shape)

        return output, attention_probs

class EfficientMultiheadAttention():
    def __init__(self,hidden_size: int = 768,num_heads: int = 1,sr_ratio: int = 1,dropout_keep_rate=0.9):
        self.attn = MultiheadAttention(hidden_size,num_heads)
        self.dropout_keep_rate = dropout_keep_rate
        #self.dropout_layer = tf.keras.layers.Dropout(rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(hidden_size,sr_ratio,strides=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self,x, hw_shape, identity=None):
        x_q = x
        print("inputs shpae:",x.shape)
        if self.sr_ratio > 1:
            x_kv = nlc_to_nhwc(x, hw_shape)
            print("after nlc to nhwc:",x_kv.shape)
            x_kv = self.sr(x_kv)
            print("after sr:",x_kv.shape)
            x_kv = nhwc_to_nlc(x_kv)
            print("after nhwc to nlc:",x_kv.shape)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.

        out = self.attn(x_q, x_kv, x_kv)[0]

        return identity + tf.nn.dropout(out, keep_prob=self.dropout_keep_rate)

class MixFFN():
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 hidden_size,
                 feedforward_channels,
                 ffn_dropout_keep_ratio=0.9,
                ):
        # self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        in_channels = hidden_size
        self.ffn_dropout_keep_ratio = ffn_dropout_keep_ratio
        # fc1 = Conv2d(
        #     in_channels=in_channels,
        #     out_channels=feedforward_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True)
        self.fc1 = tf.keras.layers.Conv2D(feedforward_channels,1,strides=1)
        # 3x3 depth wise conv to provide positional encode information
        # pe_conv = Conv2d(
        #     in_channels=feedforward_channels,
        #     out_channels=feedforward_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=(3 - 1) // 2,
        #     bias=True,
        #     groups=feedforward_channels)
        self.pe_conv = tf.keras.layers.Conv2D(feedforward_channels,3,strides=1,
                                         padding='same',groups=feedforward_channels)
        # fc2 = Conv2d(
        #     in_channels=feedforward_channels,
        #     out_channels=in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True)
        self.fc2 = tf.keras.layers.Conv2D(in_channels, 1, strides=1)
        # drop = nn.Dropout(ffn_drop)
        # self.drop = tf.keras.layers.Dropout(ffn_drop)


    def __call__(self, x, hw_shape, identity=None):
        out = nlc_to_nhwc(x, hw_shape)
        out = self.fc1(out)
        out = self.pe_conv(out)
        # out = tf.keras.activations.gelu(out)
        out = tf.keras.activations.relu(out)
        out = tf.nn.dropout(out, keep_prob=self.ffn_dropout_keep_ratio)
        out = self.fc2(out)
        out = tf.nn.dropout(out, keep_prob=self.ffn_dropout_keep_ratio)
        out = nhwc_to_nlc(out)
        if identity is None:
            identity = x
        return identity + out

class TransformerEncoderLayer():
    def __init__(self,hidden_size,
                 num_heads,
                 feedforward_channels,
                 ffn_dropout_keep_ratio=0.9,
                 dropout_keep_rate=0.9,
                 sr_ratio=1,):
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = EfficientMultiheadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_keep_rate=dropout_keep_rate,
            sr_ratio=sr_ratio)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = MixFFN(
            hidden_size=hidden_size,
            feedforward_channels=feedforward_channels,
            ffn_dropout_keep_ratio=ffn_dropout_keep_ratio,
            )
    def __call__(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x

class MixVisionTransformer():
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims=32,
                 num_stages=4,
                 num_layers=[2, 2, 2, 2],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 ffn_dropout_keep_ratio=1.0,
                 dropout_keep_rate=1.0,
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):


        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        # dpr = [
        #     x.item()
        #     for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        # ]  # stochastic num_layer decay rule
        #
        # cur = 0
        # self.layers = ModuleList()
        self.layers = []
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]

            patch_embed = PatchEmbedding(
                embed_len=embed_dims_i,
                kernel=(patch_sizes[i],patch_sizes[i]),
                stride=(strides[i],strides[i])
                )

            layer = []
            for idx in range(num_layer):
                transF = TransformerEncoderLayer(
                    hidden_size=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    ffn_dropout_keep_ratio=ffn_dropout_keep_ratio,
                    dropout_keep_rate=dropout_keep_rate,
                    sr_ratio=sr_ratios[i])
                layer.append(transF)

            # layer = ModuleList([
            #     TransformerEncoderLayer(
            #         hidden_size=embed_dims_i,
            #         num_heads=num_heads[i],
            #         feedforward_channels=mlp_ratio * embed_dims_i,
            #         drop_rate=drop_rate,
            #         attn_drop_rate=attn_drop_rate,
            #         sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            # ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            # norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            # self.layers.append(ModuleList([patch_embed, layer, norm]))
            self.layers.append([patch_embed, layer, norm])
            # cur += num_layer

    # def init_weights(self):
    #     if self.init_cfg is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 trunc_normal_init(m, std=.02, bias=0.)
    #             elif isinstance(m, nn.LayerNorm):
    #                 constant_init(m, val=1.0, bias=0.)
    #             elif isinstance(m, nn.Conv2d):
    #                 fan_out = m.kernel_size[0] * m.kernel_size[
    #                     1] * m.out_channels
    #                 fan_out //= m.groups
    #                 normal_init(
    #                     m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
    #     else:
    #         super(MixVisionTransformer, self).init_weights()

    def __call__(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            #layer: [patch_embed, layer, norm]
            x, hw_shape = layer[0](x) #patch_embed
            print("Layer {}，patch_embedding shape:{}, hw_shape:{}".format(i,x, hw_shape))
            for block in layer[1]:
                x = block(x, hw_shape)#TransformerEncoderLayer

            x = layer[2](x) #norm
            #print("x shape:{}".format(x))
            x = nlc_to_nhwc(x, hw_shape)
            print("Encoder shape:{}".format(x))
            if i in self.out_indices:
                outs.append(x)

        return outs

class MiTDecoder():
    def __init__(self,channels=256,dropout_ratio=0.1,num_classes=19):
        self.filters = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes

    def __call__(self,inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        img_resize = (inputs[0].shape[1].value*4,inputs[0].shape[2].value*4)
        net_list = []

        for x in inputs:
            net = tf.keras.layers.Conv2D(self.filters,1)(x)
            net = tf.keras.layers.LayerNormalization(epsilon=1e-6)(net)
            net = tf.keras.activations.relu(net)

            net = tf.image.resize(net,img_resize)

            net_list.append(net)

        net = tf.concat(net_list,axis=-1)

        net = tf.keras.layers.Conv2D(self.filters, 1)(net)
        net = tf.keras.layers.LayerNormalization(epsilon=1e-6)(net)
        net = tf.keras.activations.relu(net)

        net = tf.keras.layers.SpatialDropout2D(self.dropout_ratio)(net)
        net = tf.keras.layers.Conv2D(self.num_classes, 1)(net)

        return net

class MitConcatOutput():
    def __init__(self,inputs,h_min=6,print_out=False):
        # h_min = inputs[-1].shape[1].value

        out_qty = len(inputs)
        pool_times = list()
        for i in range(out_qty):
            h = inputs[i].shape[1].value
            temp = int(np.log2(h / h_min))
            pool_times.append(temp)

        #----info
        if print_out:
            print("The min output shape:",inputs[-1].shape)
            for i in range(out_qty):
                print("第{}個output shape:{}，需進行pool{}次".format(i,inputs[i].shape,pool_times[i]))

        #----
        self.inputs = inputs
        self.pool_times = pool_times
        self.out_qty = out_qty
    def __call__(self,pool_type,kernel_size,filters=64,pool_size=2,activation=tf.keras.activations.relu,
                 embed_length=128,dropout_keep_prob=None):
        net_list = []

        if pool_type == 'cnn':
            stride = 2
        else:
            stride = 1
        for i in range(self.out_qty):
            if self.pool_times[i] == 0:
                net_list.append(self.inputs[i])
            else:
                filters_c = self.inputs[i].shape[-1].value
                for pool_idx in range(self.pool_times[i]):
                    if pool_idx == 0:
                        data_input = self.inputs[i]
                    else:
                        data_input = cnn_t

                    #----cnn
                    filters_c = int(filters_c * 0.8)
                    print("filters_reduce:",filters_c)
                    cnn_t = tf.keras.layers.Conv2D(filters_c,kernel_size,strides=stride,padding='same')(data_input)
                    if activation is not None:
                        cnn_t = activation(cnn_t)

                    if pool_type == 'max':
                        cnn_t = tf.keras.layers.MaxPool2D(pool_size=pool_size)(cnn_t)
                    elif pool_type == 'ave':
                        cnn_t = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(cnn_t)

                # print(cnn_t.shape)
                net_list.append(cnn_t)

        #----cnn of last output
        # cnn_t = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(self.inputs[-1])
        # if activation is not None:
        #     cnn_t = activation(cnn_t)
        # net_list.append(cnn_t)

        net = tf.concat(net_list,axis=-1)
        print("net shape:",net.shape)

        #----
        net = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(net)
        print("net shape:", net.shape)

        #----flatten
        net = tf.keras.layers.Flatten()(net)

        if dropout_keep_prob is not None:
            net = tf.nn.dropout(net, keep_prob=dropout_keep_prob)

        net = tf.keras.layers.Dense(embed_length)(net)
        print("output shape:", net.shape)

        return net









