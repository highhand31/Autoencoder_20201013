import tensorflow
import config_mit
from models_MiT import MixVisionTransformer,PatchEmbedding,MiTDecoder

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

# b0 = {"embed_dims":32,"num_stages":4,"num_layers":[2, 2, 2, 2],"num_heads":[1, 2, 5, 8],"patch_sizes":[7, 3, 3, 3],
#       "strides":[4, 2, 2, 2],'sr_ratios':[8, 4, 2, 1],"mlp_ratio":4}
# b2 = {"embed_dims":64,"num_stages":4,"num_layers":[3, 3, 6, 3],"num_heads":[1, 2, 5, 8],"patch_sizes":[7, 3, 3, 3],
#       "strides":[4, 2, 2, 2],'sr_ratios':[8, 4, 2, 1],"mlp_ratio":4}
# config = {'b0':b0,'b2':b2}


def test_mit():
    # with pytest.raises(TypeError):
    #     # Pretrained represents pretrain url and must be str or None.
    #     MixVisionTransformer(pretrained=123)
    #----input
    dtype = 'float32'
    model_shape = [None, 544, 832, 3]
    H, W = model_shape[1:3]
    tf_input = tf.placeholder(dtype, shape=model_shape, name='input')

    #----Test patch embedding
    # emb = PatchEmbedding(kernel=(7,7),stride=(4,4),embed_len=32)
    # net, out_size = emb(tf_input)
    # print("net shape:{}, out_size:{}".format(net.shape, out_size))

    #----Test MixVisionTransformer
    name = 'b2'
    cfg = config_mit.config[name]

    num_heads = cfg['num_heads']
    embed_dims = cfg['embed_dims']
    resize_ratio = [4,8,16,32]
    model = MixVisionTransformer(
        embed_dims=cfg['embed_dims'],
        num_stages=cfg['num_stages'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        patch_sizes=cfg['patch_sizes'],
        strides=cfg['strides'],
        sr_ratios=cfg['sr_ratios'],
        mlp_ratio=cfg['mlp_ratio'])
    #model.init_weights()
    outs = model(tf_input)

    for i, out in enumerate(outs):
        real_shape = (out.shape[1].value,out.shape[2].value,out.shape[3].value)
        expected_shape = (H // resize_ratio[i], W // resize_ratio[i], num_heads[i] * embed_dims)
        assert real_shape == expected_shape, "expected_shape={}, but real_shape={}".format(expected_shape,real_shape)

    #----Test MiTDecoder
    mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],dropout_ratio=0.1,num_classes=5)
    outs = mitDec(outs)

    print(outs.shape)

    # Test non-squared input
    # H, W = (224, 256)
    # temp = torch.randn((1, 3, H, W))
    # outs = model(temp)
    # assert outs[0].shape == (1, 32, H // 4, W // 4)
    # assert outs[1].shape == (1, 64, H // 8, W // 8)
    # assert outs[2].shape == (1, 160, H // 16, W // 16)
    # assert outs[3].shape == (1, 256, H // 32, W // 32)

    # Test MixFFN
    # FFN = MixFFN(64, 128)
    # hw_shape = (32, 32)
    # token_len = 32 * 32
    # temp = torch.randn((1, token_len, 64))
    # # Self identity
    # out = FFN(temp, hw_shape)
    # assert out.shape == (1, token_len, 64)
    # # Out identity
    # outs = FFN(temp, hw_shape, temp)
    # assert out.shape == (1, token_len, 64)

    # Test EfficientMHA
    # MHA = EfficientMultiheadAttention(64, 2)
    # hw_shape = (32, 32)
    # token_len = 32 * 32
    # temp = torch.randn((1, token_len, 64))
    # # Self identity
    # out = MHA(temp, hw_shape)
    # assert out.shape == (1, token_len, 64)
    # # Out identity
    # outs = MHA(temp, hw_shape, temp)
    # assert out.shape == (1, token_len, 64)
    #
    # # Test TransformerEncoderLayer with checkpoint forward
    # block = TransformerEncoderLayer(
    #     embed_dims=64, num_heads=4, feedforward_channels=256, with_cp=True)
    # assert block.with_cp
    # x = torch.randn(1, 56 * 56, 64)
    # x_out = block(x, (56, 56))
    # assert x_out.shape == torch.Size([1, 56 * 56, 64])


def test_mit_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = MixVisionTransformer(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = MixVisionTransformer(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = MixVisionTransformer(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = MixVisionTransformer(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        MixVisionTransformer(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        MixVisionTransformer(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        MixVisionTransformer(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        MixVisionTransformer(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        MixVisionTransformer(pretrained=123, init_cfg=123)


if __name__ == "__main__":
    test_mit()