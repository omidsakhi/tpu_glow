import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer  # pylint: disable=E0611
import numpy as np

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

def default_initializer():
    return tf.variance_scaling_initializer()

def dense(name, inputs, channels, is_training, has_bn=True, init_zero=False, relu=False):
    with tf.variable_scope(name):
        inputs = tf.layers.dense(inputs, channels, bias_initializer=None, use_bias=False,
                                 kernel_initializer=default_initializer(), name=name)
        if has_bn:
            inputs = batch_norm_relu(
                "actnorm", inputs, is_training, relu=relu, init_zero=init_zero)
        return inputs


def dense_with_bias(inputs, channels, name):
    with tf.variable_scope(name):
        return tf.layers.dense(
            inputs, channels,
            bias_initializer=tf.zeros_initializer(),
            use_bias=True,
            kernel_initializer=default_initializer(),
            name=name)

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)

def channel_scale(x):
    shape = x.get_shape()
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
        _shape = (1, int_shape(x)[1])
    elif len(shape) == 4:
        _shape = (1, 1, 1, int_shape(x)[3])    
    with tf.variable_scope('ChannelScale'):        
        return x * tf.get_variable('scale', _shape, initializer=tf.zeros_initializer())

def _conv2d(name, inputs, filters, kernel_size, stride, relu=False, init_zero=False, pn=False):
    
    with tf.variable_scope(name):
        inputs = tf.layers.conv2d(
            inputs, filters, kernel_size,
            strides=[stride, stride], padding='same',
            bias_initializer=tf.zeros_initializer(),
            use_bias=True,
            kernel_initializer=default_initializer(),            
            name=name)
        if relu:
            inputs = tf.nn.relu(inputs)
        if pn:
           inputs = pixel_norm(inputs)
        if init_zero:            
            inputs = channel_scale(inputs)
        return inputs

def squeeze2d(x, factor=2):

    x = tf.space_to_depth(x, factor)
    '''
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])    
    '''
    return x


def unsqueeze2d(x, factor=2):

    x = tf.depth_to_space(x, factor)

    '''
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    '''
    return x

# Reverse features across channel dimension


def reverse_features(name, h, reverse=False):
    return h[:, :, :, ::-1]

# Shuffle across the channel dimension


def shuffle_features(name, h, indices=None, return_indices=False, reverse=False):
    with tf.variable_scope(name):

        rng = np.random.RandomState(  # pylint: disable=E1101
            (abs(hash(tf.get_variable_scope().name))) % 10000000)

        if indices == None:
            # Create numpy and tensorflow variables with indices
            n_channels = int(h.get_shape()[-1])
            indices = list(range(n_channels))
            rng.shuffle(indices)
            # Reverse it
            indices_inverse = [0]*n_channels
            for i in range(n_channels):
                indices_inverse[indices[i]] = i

        tf_indices = tf.get_variable("indices", dtype=tf.int32, initializer=np.asarray(
            indices, dtype='int32'), trainable=False)
        tf_indices_reverse = tf.get_variable("indices_inverse", dtype=tf.int32, initializer=np.asarray(
            indices_inverse, dtype='int32'), trainable=False)

        _indices = tf_indices
        if reverse:
            _indices = tf_indices_reverse

        if len(h.get_shape()) == 2:
            # Slice
            h = tf.transpose(h)
            h = tf.gather(h, _indices)
            h = tf.transpose(h)
        elif len(h.get_shape()) == 4:
            # Slice
            h = tf.transpose(h, [3, 1, 2, 0])
            h = tf.gather(h, _indices)
            h = tf.transpose(h, [3, 1, 2, 0])
        if return_indices:
            return h, indices
        return h

# Random variables


def flatten_sum(x):
    if len(x.get_shape()) == 2:
        return tf.reduce_sum(x, [1])
    elif len(x.get_shape()) == 4:
        return tf.reduce_sum(x, [1, 2, 3])
    else:
        raise Exception()


def batch_norm_relu(name, inputs, is_training, relu=True, init_zero=False):
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer, name=name)

    if relu:
        inputs = tf.nn.relu(inputs, name=name + "_relu")
    return inputs


def standard_gaussian(shape):
    return gaussian_diag(tf.zeros(shape), tf.zeros(shape))

@tf.custom_gradient
def div_by_exp(x, y):
    exp_y = tf.exp(y) + 1e-6
    ret = x / exp_y
    def _grad(dy):
        return dy/exp_y, dy*-ret
    return ret, _grad

def gaussian_diag(mean, logsd):    
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))    
    o.sample_eps = staticmethod(lambda eps: mean + tf.exp(logsd) * eps)
    o.sample = staticmethod(lambda temp: mean + tf.exp(logsd) * o.eps * temp)
    o.logps = staticmethod(lambda x: -0.5 * (np.log(2 * np.pi) +
                                             2. * logsd + div_by_exp(tf.square(x - mean),2. * logsd)))
    o.logp = staticmethod(lambda x: flatten_sum(o.logps(x)))
    o.get_eps = staticmethod(lambda x: div_by_exp(x - mean, logsd))
    return o


# def discretized_logistic_old(mean, logscale, binsize=1 / 256.0, sample=None):
#    scale = tf.exp(logscale)
#    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
#    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
#    return tf.reduce_sum(logp, [1, 2, 3])
def discretized_logistic(mean, logscale, binsize=1. / 256):
    class o(object):
        pass
    o.mean = mean
    o.logscale = logscale
    scale = tf.exp(logscale)

    def logps(x):
        x = (x - mean) / scale
        return tf.log(tf.sigmoid(x + binsize / scale) - tf.sigmoid(x) + 1e-7)
    o.logps = logps
    o.logp = staticmethod(lambda x: flatten_sum(logps(x)))
    return o


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(
        tf.matmul(u, tf.diag(si)), v, transpose_b=True)


def scale_bias(name, x, scale_factor=1., logdet=None, logscale_factor=3., reverse=False):
    with tf.variable_scope(name):
        if not reverse:
            x = bias("bias", x, reverse)
            x = scale("scale", x, scale_factor,
                      logdet, logscale_factor, reverse)
            if logdet != None:
                x, logdet = x
        else:
            x = scale("scale", x, scale_factor,
                      logdet, logscale_factor, reverse)
            if logdet != None:
                x, logdet = x
            x = bias("bias", x, reverse)
        if logdet != None:
            return x, logdet
        return x


def bias(name, x, reverse=False):
    shape = x.get_shape()
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
        _shape = (1, int_shape(x)[1])
    elif len(shape) == 4:
        _shape = (1, 1, 1, int_shape(x)[3])
    b = tf.get_variable(name, _shape, dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    if not reverse:
        x += b
    else:
        x -= b
    return x


def scale(name, x, scale=1., logdet=None, logscale_factor=3., reverse=False):
    shape = x.get_shape()
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
        _shape = (1, int_shape(x)[1])
        logdet_factor = 1
    elif len(shape) == 4:
        _shape = (1, 1, 1, int_shape(x)[3])
        logdet_factor = int(shape[1])*int(shape[2])
    logs = tf.get_variable(name, _shape, initializer=tf.zeros_initializer()) / 4.0
    if not reverse:
        x *= tf.exp(logs)
    else:
        x *= tf.exp(-1.0 * logs)
    if logdet != None:
        dlogdet =  tf.reduce_sum(logs) * logdet_factor
        if reverse:
            dlogdet *= -1
        return x, logdet + dlogdet
    return x    
