import ops
import numpy as np
import tensorflow as tf


def codec(cfg):

    def encoder(z, objective, is_training):        
        eps = []
        for i in range(cfg.n_levels):
            z, objective = revnet2d(str(i), z, objective, cfg, is_training=is_training)            
            if i < cfg.n_levels-1:
                z, objective, _eps = split2d("pool"+str(i), z, objective=objective)                
                eps.append(_eps)            
        z = tf.identity(z, name = 'z')
        return z, objective, eps

    def decoder(z, is_training, eps=[None]*cfg.n_levels, eps_std=None):        
        for i in reversed(range(cfg.n_levels)):
            if i < cfg.n_levels-1:
                z = split2d_reverse("pool"+str(i), z, eps=eps[i], eps_std=eps_std)
            z, _ = revnet2d(str(i), z, 0, cfg, is_training=is_training, reverse=True)            
        return z

    return encoder, decoder

def revnet2d(name, z, logdet, cfg, is_training, reverse=False):
    with tf.variable_scope(name):
        if not reverse:
            for i in range(cfg.depth):
                z, logdet = checkpoint(z, logdet)
                z, logdet = revnet2d_step(str(i), z, logdet, cfg, reverse, is_training)
            z, logdet = checkpoint(z, logdet)
        else:
            for i in reversed(range(cfg.depth)):
                z, logdet = revnet2d_step(str(i), z, logdet, cfg, reverse, is_training)
    return z, logdet

# Simpler, new version
def revnet2d_step(name, z, logdet, cfg, reverse, is_training):
    with tf.variable_scope(name):

        shape = ops.int_shape(z)
        n_z = shape[3]
        assert n_z % 2 == 0

        if not reverse:

            #z, logdet = ops.actnorm_("actnorm", z, logdet=logdet, is_training=is_training)

            if cfg.flow_permutation == 0:
                z = ops.reverse_features("reverse", z)
            elif cfg.flow_permutation == 1:
                z = ops.shuffle_features("shuffle", z)
            elif cfg.flow_permutation == 2:
                z, logdet = invertible_1x1_conv("invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if cfg.flow_coupling == 0:
                z2 += f_("f1", z1, cfg.width, is_training=is_training)
            elif cfg.flow_coupling == 1:
                h = f_("f1", z1, cfg.width, n_z, is_training=is_training)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

        else:

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if cfg.flow_coupling == 0:
                z2 -= f_("f1", z1, cfg.width, is_training=is_training)
            elif cfg.flow_coupling == 1:
                h = f_("f1", z1, cfg.width, n_z, is_training=is_training)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

            if cfg.flow_permutation == 0:
                z = ops.reverse_features("reverse", z, reverse=True)
            elif cfg.flow_permutation == 1:
                z = ops.shuffle_features("shuffle", z, reverse=True)
            elif cfg.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(
                    "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            #z, logdet = ops.actnorm_("actnorm", z, logdet=logdet, reverse=True, is_training=is_training)

    return z, logdet

def f_(name, h, width, n_out=None, is_training = False):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(ops.conv2d_("l_1", h, width, is_training=is_training))
        h = tf.nn.relu(ops.conv2d_("l_2", h, width, filter_size=[1, 1], is_training=is_training))
        h = ops.conv2d_zeros("l_last", h, n_out)
    return h

def split2d(name, z, objective=0.):
    with tf.variable_scope(name):
        n_z = ops.int_shape(z)[3]
        z1 = z[:, :, :, :n_z // 2]
        z2 = z[:, :, :, n_z // 2:]
        pz = split2d_prior(z1)
        objective += pz.logp(z2)
        z1 = ops.squeeze2d(z1)
        eps = pz.get_eps(z2)
        return z1, objective, eps


def split2d_reverse(name, z, eps, eps_std):
    with tf.variable_scope(name):
        z1 = ops.unsqueeze2d(z)
        pz = split2d_prior(z1)
        if eps is not None:
            # Already sampled eps
            z2 = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z2 = pz.sample
        z = tf.concat([z1, z2], 3)
        return z

def split2d_prior(z):
    n_z2 = int(z.get_shape()[3])
    n_z1 = n_z2
    h = ops.conv2d_zeros("conv", z, 2 * n_z1)

    mean = h[:, :, :, 0::2]
    logs = h[:, :, :, 1::2]
    return ops.gaussian_diag(mean, logs)

def checkpoint(z, logdet):
    zshape = ops.int_shape(z)
    z = tf.reshape(z, [-1, zshape[1]*zshape[2]*zshape[3]])
    logdet = tf.reshape(logdet, [-1, 1])
    combined = tf.concat([z, logdet], axis=1)
    tf.add_to_collection('checkpoints', combined)
    logdet = combined[:, -1]
    z = tf.reshape(combined[:, :-1], [-1, zshape[1], zshape[2], zshape[3]])
    return z, logdet

# Invertible 1x1 conv
def invertible_1x1_conv(name, z, logdet, reverse=False):

    if True:  # Set to "False" to use the LU-decomposed version

        with tf.variable_scope(name):

            shape = ops.int_shape(z)
            w_shape = [shape[3], shape[3]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init) + tf.eye(shape[3]) * 10e-4            

            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]

            if not reverse:

                _w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += dlogdet

                return z, logdet
            else:

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1]+w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet -= dlogdet

                return z, logdet

    else:

        # LU-decomposed version
        shape = ops.int_shape(z)
        with tf.variable_scope(name):

            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w) # pylint: disable=E1101
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[3], shape[3]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet

def prior(name, y_onehot, cfg):

    with tf.variable_scope(name):
                
        cfg.top_shape = [4,4, 192]

        n_z = cfg.top_shape[-1]

        h = tf.zeros([tf.shape(y_onehot)[0]]+cfg.top_shape[:2]+[2*n_z])
        if cfg.learntop:
            h = ops.conv2d_zeros('p', h, 2*n_z)
        if cfg.ycond:
            h += tf.reshape(ops.linear_zeros("y_emb", y_onehot,
                                           2*n_z), [-1, 1, 1, 2 * n_z])

        pz = ops.gaussian_diag(h[:, :, :, :n_z], h[:, :, :, n_z:])

    def logp(z1):
        objective = pz.logp(z1)
        return objective

    def sample(eps=None, eps_std=None):
        if eps is not None:
            # Already sampled eps. Don't use eps_std
            z = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z = pz.sample

        return z

    def eps(z1):
        return pz.get_eps(z1)

    return logp, sample, eps

class model(object):
    cfg = None
    encoder = None
    decoder = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder, self.decoder = codec(cfg)    
        self.cfg.n_bins = 2. ** self.cfg.n_bits_x

    def _f_loss(self, x, y, is_training):

        with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
            y_onehot = tf.cast(tf.one_hot(y, self.cfg.n_y, 1, 0), 'float32')

            # Discrete -> Continuous
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]            
            z = x + tf.random_uniform(tf.shape(x), 0, 1./self.cfg.n_bins)
            objective += - np.log(self.cfg.n_bins) * np.prod(ops.int_shape(z)[1:])

            # Encode
            z = ops.squeeze2d(z, 2)  # > 16x16x12
            z, objective, _ = self.encoder(z, objective, is_training)                        

            # Prior
            self.cfg.top_shape = ops.int_shape(z)[1:]
            logp, _, _ = prior("prior", y_onehot, self.cfg)            

            objective += logp(z)            

            # Generative loss
            nobj = - objective
            bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(
                x.get_shape()[2]) * int(x.get_shape()[3]))  # bits per subpixel

            # Predictive loss
            if self.cfg.weight_y > 0 and self.cfg.ycond:

                # Classification loss
                h_y = tf.reduce_mean(z, axis=[1, 2])
                y_logits = ops.linear_zeros("classifier", h_y, self.cfg.n_y)
                bits_y = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_onehot, logits=y_logits) / np.log(2.)

                # Classification accuracy
                y_predicted = tf.argmax(y_logits, 1, output_type=tf.int32)
                classification_error = 1 - \
                    tf.cast(tf.equal(y_predicted, y), tf.float32)
            else:
                bits_y = tf.zeros_like(bits_x)
                classification_error = tf.ones_like(bits_x)

        return bits_x, bits_y, classification_error

    def f_loss(self,x, y, is_training):
        bits_x, bits_y, pred_loss = self._f_loss(x, y, is_training)
        local_loss = bits_x + self.cfg.weight_y * bits_y
        stats = [local_loss, bits_x, bits_y, pred_loss]
        global_stats = ops.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats]))
        return tf.reduce_mean(local_loss), global_stats

    # === Sampling function
    def sample(self, y, is_training):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            y_onehot = tf.cast(tf.one_hot(y, self.cfg.n_y, 1, 0), 'float32')
            _, sample, _ = prior("prior", y_onehot, self.cfg)
            z = sample()
            x = self.decoder(z, is_training)
            x = ops.unsqueeze2d(x, 2)  # 8x8x12 -> 16x16x3    
            x = self.postprocess(x)        
        return x

    def postprocess(self,x):
        return tf.cast(tf.clip_by_value(tf.floor((x + .5)*self.cfg.n_bins)*(256./self.cfg.n_bins), 0, 255), 'uint8')

    def f_encode(self, x, y, is_training):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            y_onehot = tf.cast(tf.one_hot(y, self.cfg.n_y, 1, 0), 'float32')

            # Discrete -> Continuous
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]            
            z = x + tf.random_uniform(tf.shape(x), 0, 1. / self.cfg.n_bins)
            objective += - np.log(self.cfg.n_bins) * np.prod(ops.int_shape(z)[1:])

            # Encode
            z = ops.squeeze2d(z, 2)  # > 16x16x12
            z, objective, eps = self.encoder(z, objective, is_training)

            # Prior
            self.cfg.top_shape = ops.int_shape(z)[1:]
            logp, _, _eps = prior("prior", y_onehot, self.cfg)
            objective += logp(z)
            eps.append(_eps(z))
        return eps

    def f_decode(self, y, eps, is_training):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            y_onehot = tf.cast(tf.one_hot(y, self.cfg.n_y, 1, 0), 'float32')
            _, sample, _ = prior("prior", y_onehot, self.cfg)
            z = sample(eps=eps[-1])
            z = self.decoder(z, is_training = is_training, eps=eps[:-1])
            z = ops.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3     
            x = self.postprocess(z)       
        return x