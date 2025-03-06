"""
import tensorflow as tf
from sam2.build_sam import build_sam2

class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        return self.double_conv(x)


class Up(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def call(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]
        x1 = tf.pad(x1, [[0, 0], [diffY // 2, diffY - diffY // 2], [diffX // 2, diffX - diffX // 2], [0, 0]])
        x = tf.concat([x2, x1], axis=3)
        return self.conv(x)


class Adapter(tf.keras.layers.Layer):
    def __init__(self, blk):
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='gelu'),
            tf.keras.layers.Dense(dim, activation='gelu')
        ])

    def call(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(tf.keras.layers.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding='same' if padding else 'valid', dilation_rate=dilation, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.branch0 = tf.keras.models.Sequential([
            BasicConv2d(in_channel, out_channel, 1)
        ])
        self.branch1 = tf.keras.models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        ])
        self.branch2 = tf.keras.models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        ])
        self.branch3 = tf.keras.models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        ])
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def call(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(tf.concat([x0, x1, x2, x3], axis=3))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class SAM2UNet(tf.keras.Model):
    def __init__(self, checkpoint_path=None):
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2_hiera_t.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Adapter(block))
        self.encoder.blocks = tf.keras.models.Sequential(blocks)
        self.rfb1 = RFB_modified(96, 64)
        self.rfb2 = RFB_modified(192, 64)
        self.rfb3 = RFB_modified(384, 64)
        self.rfb4 = RFB_modified(768, 64)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)
        self.side1 = tf.keras.layers.Conv2D(64, 1)
        self.side2 = tf.keras.layers.Conv2D(64, 1)
        self.head = tf.keras.layers.Conv2D(64, 1)

    def call(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = tf.image.resize(self.side1(x), size=(self.input_shape[1]//16, self.input_shape[2]//16), method='bilinear')
        x = self.up2(x, x2)
        out2 = tf.image.resize(self.side2(x), size=(self.input_shape[1]//8, self.input_shape[2]//8), method='bilinear')
        x = self.up3(x, x1)
        out = tf.image.resize(self.head(x), size=(self.input_shape[1]//4, self.input_shape[2]//4), method='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    model = SAM2UNet()
    x = tf.random.normal([1, 352, 352, 3])
    out, out1, out2 = model(x)
    print(out.shape, out1.shape, out2.shape)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from sam2.build_sam import build_sam2

class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = models.Sequential([
            layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        return self.double_conv(x)

class Up(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def call(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]
        x1 = tf.pad(x1, [[0, 0], [diffY // 2, diffY - diffY // 2], [diffX // 2, diffX - diffX // 2], [0, 0]])
        x = tf.concat([x2, x1], axis=3)
        return self.conv(x)

class Adapter(tf.keras.layers.Layer):
    def __init__(self, blk):
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = models.Sequential([
            layers.Dense(32, activation='gelu'),
            layers.Dense(dim, activation='gelu')
        ])

    def call(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class BasicConv2d(tf.keras.layers.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding='same' if padding else 'valid', dilation_rate=dilation, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = layers.ReLU()
        self.branch0 = models.Sequential([
            BasicConv2d(in_channel, out_channel, 1)
        ])
        self.branch1 = models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding='same'),
            BasicConv2d(out_channel, out_channel, (3, 1), padding='same'),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        ])
        self.branch2 = models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 5), padding='same'),
            BasicConv2d(out_channel, out_channel, (5, 1), padding='same'),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        ])
        self.branch3 = models.Sequential([
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 7), padding='same'),
            BasicConv2d(out_channel, out_channel, (7, 1), padding='same'),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        ])
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding='same')
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def call(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(tf.concat([x0, x1, x2, x3], axis=3))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class SAM2UNet(tf.keras.Model):
    def __init__(self, checkpoint_path=None):
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2_hiera_t.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Wrap the encoder blocks in Adapters
        self.encoder_blocks = models.Sequential([Adapter(block) for block in self.encoder.blocks])
        
        self.rfb1 = RFB_modified(96, 64)
        self.rfb2 = RFB_modified(192, 64)
        self.rfb3 = RFB_modified(384, 64)
        self.rfb4 = RFB_modified(768, 64)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)
        self.side1 = layers.Conv2D(1, 1)
        self.side2 = layers.Conv2D(1, 1)
        self.head = layers.Conv2D(1, 1)

    def call(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = tf.image.resize(self.side1(x), (x.shape[1] * 16, x.shape[2] * 16), method='bilinear')
        x = self.up2(x, x2)
        out2 = tf.image.resize(self.side2(x), (x.shape[1] * 8, x.shape[2] * 8), method='bilinear')
        x = self.up3(x, x1)
        out = tf.image.resize(self.head(x), (x.shape[1] * 4, x.shape[2] * 4), method='bilinear')
        return out, out1, out2

if __name__ == "__main__":
    model = SAM2UNet()
    x = tf.random.normal([1, 352, 352, 3])
    out, out1, out2 = model(x)
    print(out.shape, out1.shape, out2.shape)