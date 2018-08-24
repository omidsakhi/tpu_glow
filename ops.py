import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer # pylint: disable=E0611
import numpy as np

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def allreduce_sum(x):
    return tf.reduce_sum(x)

def allreduce_mean(x):
    return tf.reduce_mean(x)

def default_initial_value(shape, std=0.01):
    return tf.random_normal(shape, 0., std)

def default_initializer(std=0.01):
    return tf.random_normal_initializer(0., std)

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init

def get_variable_ddi_(name, shape, initial_value, is_training, dtype=tf.float32):
    w = tf.get_variable(name, shape, dtype, None, trainable=is_training)
    if is_training:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

# Activation normalization
# Convenience function that does centering+scaling

def actnorm_(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, is_training = False):
    if not reverse:
        x = actnorm_center_(name+"_center", x, reverse=reverse, is_training=is_training)
        x = actnorm_scale_(name+"_scale", x, scale, logdet, logscale_factor, batch_variance, reverse=reverse, is_training=is_training)
        if logdet != None:
            x, logdet = x
    else:
        x = actnorm_scale_(name + "_scale", x, scale, logdet, logscale_factor, batch_variance, reverse = reverse, is_training=is_training)
        if logdet != None:
            x, logdet = x
        x = actnorm_center_(name+"_center", x, reverse=reverse, is_training=is_training)
    if logdet != None:
        return x, logdet
    return x

# Activation normalization
def actnorm_center_(name, x, is_training, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = None
            if is_training:
                x_mean = -1.0 * tf.reduce_mean(x, [0], keepdims=True)                
            b = get_variable_ddi_("b", (1, int_shape(x)[1]), initial_value=x_mean, is_training=is_training and not reverse)
        elif len(shape) == 4:
            x_mean = None
            if is_training:
                x_mean = -1.0 * tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi_("b", (1, 1, 1, int_shape(x)[3]), initial_value=x_mean, is_training=is_training and not reverse)
        if not reverse:
            x += b
        else:
            x -= b

        return x

# Activation normalization
def actnorm_scale_(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, is_training = False):
    shape = x.get_shape()
    with tf.variable_scope(name): # pylint: disable=E1129
        assert len(shape) == 2 or len(shape) == 4
        
        if len(shape) == 2:            
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:            
            logdet_factor = int(shape[1])*int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])


        x_var = None
        init_val = None
        if is_training:
            if len(shape) == 2:
                x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            elif len(shape) == 4:
                x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
            if batch_variance:
                x_var = tf.reduce_mean(x**2, keepdims=True)
            init_val = tf.log(scale/(tf.sqrt(x_var)+1e-6))/logscale_factor
        
        logs = get_variable_ddi_("logs", _shape, initial_value=init_val, is_training=is_training and not reverse)*logscale_factor
        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)

        #s = get_variable_ddi("s", _shape, initial_value=scale / (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
        #logs = tf.log(tf.abs(s))
        #if not reverse:
        #    x *= s
        #else:
        #    x /= s

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x

# Linear layer with layer norm
def linear_(name, x, width, do_weightnorm=True, do_actnorm=True, initializer=None, scale=1., is_training = False):
    initializer = initializer or default_initializer()
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width],
                            tf.float32, initializer=initializer)
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0])
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        if do_actnorm:
            x = batch_norm_relu("actnorm", x, is_training, relu=False)            
        return x

# Linear layer with zero init
def linear_zeros(name, x, width, logscale_factor=3):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=default_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        return x

# Slow way to add edge padding


def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    if True:
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        ix = int_shape(x)[1:3]        
        name = "_".join([str(dim) for dim in [a, b, ix[0], ix[1] ]])        
        pads = tf.get_collection(name)
        if not pads:
            pad = np.zeros([1] + int_shape(x)[1:3] + [1], dtype='float32')
            pad[:, :a, :, 0] = 1.
            pad[:, -a:, :, 0] = 1.
            pad[:, :, :b, 0] = 1.
            pad[:, :, -b:, 0] = 1.
            pad = tf.convert_to_tensor(pad)
            tf.add_to_collection(name, pad)
        else:
            pad = pads[0]
        pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
        x = tf.concat([x, pad], axis=3)
    else:
        pad = tf.pad(tf.zeros_like(x[:, :, :, :1]) - 1,
                     [[0, 0], [a, a], [b, b], [0, 0]]) + 1
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        x = tf.concat([x, pad], axis=3)
    return x


def conv2d_(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", do_weightnorm=False, do_actnorm=True, context1d=None, edge_bias=True, is_training = False):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        
        x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')

        if do_actnorm:            
            x = batch_norm_relu("actnorm", x, is_training, relu=False)            
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer())

        if context1d != None:
            x += tf.reshape(linear_("context", context1d, width, is_training=is_training), [-1, 1, 1, width])
    return x

def separable_conv2d_(name, x, width, filter_size=[3, 3], stride=[1, 1], padding="SAME", do_actnorm=True, std=0.05, is_training = False):
    n_in = int(x.get_shape()[3])
    with tf.variable_scope(name):
        assert filter_size[0] % 2 == 1 and filter_size[1] % 2 == 1
        strides = [1] + stride + [1]
        w1_shape = filter_size + [n_in, 1]
        w1_init = np.zeros(w1_shape, dtype='float32')
        w1_init[(filter_size[0]-1)//2, (filter_size[1]-1)//2, :,
                :] = 1.  # initialize depthwise conv as identity
        w1 = tf.get_variable("W1", dtype=tf.float32, initializer=w1_init)
        w2_shape = [1, 1, n_in, width]
        w2 = tf.get_variable("W2", w2_shape, tf.float32,
                             initializer=default_initializer(std))
        x = tf.nn.separable_conv2d(
            x, w1, w2, strides, padding, data_format='NHWC')
        if do_actnorm:
            x = batch_norm_relu("actnorm", x, is_training, relu=False)            
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer(std))

    return x

def conv2d_zeros(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", logscale_factor=3, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.zeros_initializer())        
    return x


# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
def upsample2d_nearest_neighbour(x):
    shape = x.get_shape()
    n_batch = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    x = tf.reshape(x, (n_batch, height, 1, width, 1, n_channels))
    x = tf.concat(2, [x, x])
    x = tf.concat(4, [x, x])
    x = tf.reshape(x, (n_batch, height*2, width*2, n_channels))
    return x


def upsample(x, factor=2):
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    x = tf.image.resize_nearest_neighbor(x, [height * factor, width * factor])
    return x


def squeeze2d(x, factor=2):
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
    return x


def unsqueeze2d(x, factor=2):
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
    return x

# Reverse features across channel dimension


def reverse_features(name, h, reverse=False):
    return h[:, :, :, ::-1]

# Shuffle across the channel dimension


def shuffle_features(name, h, indices=None, return_indices=False, reverse=False):
    with tf.variable_scope(name):

        rng = np.random.RandomState( # pylint: disable=E1101
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


def embedding(name, y, n_y, width):
    with tf.variable_scope(name):
        params = tf.get_variable(
            "embedding", [n_y, width], initializer=default_initializer())
        embeddings = tf.gather(params, y)
        return embeddings

# Random variables


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
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
        gamma_initializer=gamma_initializer, name = name)

    if relu:
        inputs = tf.nn.relu(inputs, name = name + "_relu")
    return inputs

def standard_gaussian(shape):
    return gaussian_diag(tf.zeros(shape), tf.zeros(shape))


def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))
    o.sample = mean + tf.exp(logsd) * o.eps
    o.sample2 = staticmethod(lambda eps: mean + tf.exp(logsd) * eps)
    o.logps = staticmethod(lambda x: -0.5 * (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd) ))
    o.logp = staticmethod(lambda x: flatten_sum(o.logps(x)))
    o.get_eps = staticmethod(lambda x: (x - mean) / tf.exp(logsd))
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

class ConvolutionalBatchNormalizer(object):
  """Helper class that groups the normalization logic and variables.        

  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
      update_assignments = bn.get_assigner()                                
      x = bn.normalize(y, train=training?)                                  
      (the output x will be batch-normalized).                              
  """

  def __init__(self, depth, axis, epsilon, ewma_trainer, scale_after_norm):
    self.axis = axis
    self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                            trainable=False)
    self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                trainable=False)
    self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
    self.ewma_trainer = ewma_trainer
    self.epsilon = epsilon
    self.scale_after_norm = scale_after_norm

  def get_assigner(self):
    """Returns an EWMA apply op that must be invoked after optimization."""
    return self.ewma_trainer.apply([self.mean, self.variance])

  def normalize(self, x, train=True):
    """Returns a batch-normalized version of x."""
    if train:
      mean, variance = tf.nn.moments(x, self.axis)
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(variance)
      with tf.control_dependencies([assign_mean, assign_variance]):
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, self.beta, self.gamma,
            self.epsilon, self.scale_after_norm)
    else:
      mean = self.ewma_trainer.average(self.mean)
      variance = self.ewma_trainer.average(self.variance)
      local_beta = tf.identity(self.beta)
      local_gamma = tf.identity(self.gamma)
      return tf.nn.batch_norm_with_global_normalization(
          x, mean, variance, local_beta, local_gamma,
          self.epsilon, self.scale_after_norm)