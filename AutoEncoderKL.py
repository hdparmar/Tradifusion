import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from layers import PaddedConv2D, apply_seq


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(channels, 1)
        self.k = PaddedConv2D(channels, 1)
        self.v = PaddedConv2D(channels, 1)
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute attention
        b, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b,hw,c
        k = keras.layers.Permute((3, 1, 2))(k)
        k = tf.reshape(k, (-1, c, h * w))  # b,c,hw
        w_ = q @ k
        w_ = w_ * (c ** (-0.5))
        w_ = keras.activations.softmax(w_)

        # Attend to values
        v = keras.layers.Permute((3, 1, 2))(v)
        v = tf.reshape(v, (-1, c, h * w))
        w_ = keras.layers.Permute((2, 1))(w_)
        h_ = v @ w_
        h_ = keras.layers.Permute((2, 1))(h_)
        h_ = tf.reshape(h_, (-1, h, w, c))
        return x + self.proj_out(h_)


class ResnetBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(out_channels, 3, padding=1)
        self.norm2 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(out_channels, 3, padding=1)
        self.nin_shortcut = (
            PaddedConv2D(out_channels, 1)
            if in_channels != out_channels
            else lambda x: x
        )

    def call(self, x):
        h = self.conv1(keras.activations.swish(self.norm1(x)))
        h = self.conv2(keras.activations.swish(self.norm2(h)))
        return self.nin_shortcut(x) + h
    

class CustomEncoder(keras.Model):
    def __init__(self):
        super(CustomEncoder, self).__init__()

        self.padded_conv1 = PaddedConv2D(128, 3, padding=1)
        self.res1 = ResnetBlock(128, 128)
        self.res2 = ResnetBlock(128, 128)
        self.strided1 = PaddedConv2D(128, 3, padding=(0,1), stride=2)

        self.res3 = ResnetBlock(128, 256)
        self.res4 = ResnetBlock(256, 256)
        self.strided2 = PaddedConv2D(256, 3, padding=(0,1), stride=2)

        self.res5 = ResnetBlock(256, 512)
        self.res6 = ResnetBlock(512, 512)
        self.strided3 = PaddedConv2D(512, 3, padding=(0,1), stride=2)

        self.res7 = ResnetBlock(512, 512)
        self.res8 = ResnetBlock(512, 512)

        self.res9 = ResnetBlock(512, 512)
        self.attention = AttentionBlock(512)
        self.res10 = ResnetBlock(512, 512)

        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.activation = keras.layers.Activation("swish")
        self.final_conv = PaddedConv2D(8, 3, padding=1)
        self.final_conv2 = PaddedConv2D(8, 1)
        self.lambda_layer = keras.layers.Lambda(lambda x : x[... , :4] * 0.18215)

        # These are the new layers for KL divergence
        self.mu = keras.layers.Conv2D(4, 1, padding='same')
        self.log_var = keras.layers.Conv2D(4, 1, padding='same')

    def call(self, x):
        x = self.padded_conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.strided1(x)

        x = self.res3(x)
        x = self.res4(x)
        x = self.strided2(x)

        x = self.res5(x)
        x = self.res6(x)
        x = self.strided3(x)

        x = self.res7(x)
        x = self.res8(x)

        x = self.res9(x)
        x = self.attention(x)
        x = self.res10(x)

        x = self.norm(x)
        x = self.activation(x)
        x = self.final_conv(x)
        x = self.final_conv2(x)
        x = self.lambda_layer(x)

        # These are the new parts for KL divergence
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        epsilon = tf.random.normal(shape=mu.shape)
        z = mu + tf.exp(0.5 * log_var) * epsilon
        return z

class CustomDecoder(keras.Model):
    def __init__(self):
        super(CustomDecoder, self).__init__()

        self.deconv_start = PaddedConv2D(4, 1) 

        # First set of res blocks and upsampling
        self.upconv1 = PaddedConv2D(512, 3, padding=1)
        self.res1 = ResnetBlock(512, 512)
        self.att1 = AttentionBlock(512)
        self.res2 = ResnetBlock(512, 512)

        # Second set
        self.upconv2 = keras.layers.UpSampling2D(size=(2, 2))
        self.res3 = ResnetBlock(512, 256)
        self.res4 = ResnetBlock(256, 256)

        # Third set
        self.upconv3 = keras.layers.UpSampling2D(size=(2, 2))
        self.res5 = ResnetBlock(256, 128)
        self.res6 = ResnetBlock(128, 128)

        self.upconv4 = keras.layers.UpSampling2D(size=(2, 2))
        self.res7 = ResnetBlock(128, 64)
        self.res8 = ResnetBlock(64, 64)

        # Final touches
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.act = keras.layers.Activation("swish")
        self.final_conv = PaddedConv2D(3, 3, padding=1)

    def call(self, z):
        x = self.deconv_start(z)

        x = self.upconv1(x)
        x = self.res1(x)
        x = self.att1(x)
        x = self.res2(x)

        x = self.upconv2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.upconv3(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.upconv4(x)
        x = self.res7(x)
        x = self.res8(x)

        x = self.norm(x)
        x = self.act(x)
        x = self.final_conv(x)

        return x


# AutoencoderKL Class
class AutoencoderKL(tf.keras.Model):
    def __init__(self):
        super(AutoencoderKL, self).__init__()
        self.encoder = CustomEncoder()
        self.decoder = CustomDecoder()

    def call(self, x):
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
    
# KL Divergence Loss Function 
def kl_divergence_loss(mu, log_var):
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return kl_loss

# Reconstruction Loss
def reconstruction_loss(x, x_reconstructed):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(x, x_reconstructed)