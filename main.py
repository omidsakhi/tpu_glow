import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.tpu.python.tpu import tpu_config  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_estimator  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer  # pylint: disable=E0611
from tensorflow.python.estimator import estimator  # pylint: disable=E0611
import math
import ops
import memory_saving_gradients
import celeba


global dataset
dataset = celeba

USE_TPU = False

import models


def model_fn(features, labels, mode, params):

    del labels

    cfg = params['cfg']
    model = models.model(cfg)
    y = features['y']

    if mode == tf.estimator.ModeKeys.PREDICT:
        ###########
        # PREDICT #
        ###########
        predictions = {
            'generated_images': model.sample(y, is_training=False, temp=0.75)
        }
        return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    real_images = features['real_images']

    f_loss, eps = model.f_loss(real_images, y, is_training)

    if mode == tf.estimator.ModeKeys.TRAIN:
        #########
        # TRAIN #
        #########

        f_loss = tf.reduce_mean(f_loss)

        with tf.variable_scope('Regularization'):
            for v in tf.trainable_variables():
                if 'invw' in v.name:
                    det = tf.matrix_determinant(v * tf.transpose(v))
                    f_loss += 0.001 * tf.square(det)
                    f_loss -= det

            if cfg.use_l2_regularization:
                for v in tf.trainable_variables():
                    if 'actnorm' not in v.name:
                        f_loss += cfg.l2_regularization_factor * tf.nn.l2_loss(v)

        if not cfg.use_tpu and cfg.report_histograms:
            for v in tf.trainable_variables():
                tf.summary.histogram(v.name.replace(':', '_'), v)

        #lr = int(real_images.get_shape()[0]) * cfg.lr
        lr = cfg.lr
        from AMSGrad import AMSGrad
        optimizer = AMSGrad(
            learning_rate=lr, beta1=cfg.beta1, epsilon=cfg.adam_eps)        

        if cfg.use_tpu:
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('TrainOps'):
                if cfg.memory_saving_gradients:
                    from memory_saving_gradients import gradients
                    gs = gradients(f_loss, tf.trainable_variables())
                else:
                    gs = tf.gradients(f_loss, tf.trainable_variables())
                if cfg.use_gradient_clipping:
                    gs = [tf.clip_by_value(g, -100., 100.) for g in gs]
                grads_and_vars = list(zip(gs, tf.trainable_variables()))
                train_op = optimizer.apply_gradients(grads_and_vars)
                increment_step = tf.assign_add(
                    tf.train.get_or_create_global_step(), 1)
                joint_op = tf.group([train_op, increment_step])

            return tpu_estimator.TPUEstimatorSpec(
                mode=mode,
                loss=f_loss,
                train_op=joint_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        ########
        # EVAL #
        ########
        def _eval_metric_fn(f_loss):
            return {
                'f_loss': tf.metrics.mean(f_loss)}
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=tf.reduce_mean(f_loss),
            eval_metrics=(_eval_metric_fn, [f_loss]))

    raise ValueError('Invalid mode provided to model_fn')


def y_input_fn(params):
    batch_size = params['batch_size']
    np.random.seed(0)
    y = tf.constant(np.zeros((batch_size, 1, 1)), dtype=tf.int32)
    y = tf.data.Dataset.from_tensor_slices(y)
    y = y.batch(batch_size)
    y = y.make_one_shot_iterator().get_next()
    return {'y': y}, None


def write_images(images, filename):
    sq = math.floor(math.sqrt(len(images)))
    assert sq ** 2 == len(images)
    sq = int(sq)
    image_rows = [np.concatenate(images[i:i+sq], axis=0)
                  for i in range(0, len(images), sq)]
    tiled_image = np.concatenate(image_rows, axis=1)
    img = Image.fromarray(tiled_image, mode='RGB')
    file_obj = tf.gfile.Open(filename, 'w')
    img.save(file_obj, format='png')


def main(cfg):
    tpu_cluster_resolver = None

    if cfg.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            cfg.tpu,
            zone=cfg.tpu_zone,
            project=cfg.gcp_project)

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=cfg.model_dir,
        tpu_config=tpu_config.TPUConfig(
            num_shards=cfg.num_shards,
            iterations_per_loop=cfg.iterations_per_loop))

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        use_tpu=cfg.use_tpu,
        config=config,
        params={"cfg": cfg, "data_dir": cfg.data_dir},
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size)

    local_est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        use_tpu=False,
        config=config,
        params={"cfg": cfg, "data_dir": cfg.data_dir},
        predict_batch_size=cfg.num_eval_images)

    if cfg.mode == 'train':
        tf.gfile.MakeDirs(os.path.join(cfg.model_dir))
        tf.gfile.MakeDirs(os.path.join(cfg.model_dir, 'generated_images'))

        current_step = estimator._load_global_step_from_checkpoint_dir(
            cfg.model_dir)   # pylint: disable=protected-access,line-too-long
        tf.logging.info('Starting training for %d steps, current step: %d' % (
            cfg.train_steps, current_step))
        while current_step < cfg.train_steps:
            next_checkpoint = min(
                current_step + cfg.train_steps_per_eval, cfg.train_steps)
            est.train(input_fn=dataset.InputFunction(
                True), max_steps=next_checkpoint)
            current_step = next_checkpoint
            tf.logging.info('Finished training step %d' % current_step)

            #metrics = est.evaluate(input_fn=dataset.InputFunction(False), steps=cfg.num_eval_images // cfg.batch_size)
            #tf.logging.info('Finished evaluating')
            # tf.logging.info(metrics)

            generated_iter = local_est.predict(input_fn=y_input_fn)
            images = []
            images = [p['generated_images'][:, :, :] for p in generated_iter]
            filename = os.path.join(
                cfg.model_dir, 'generated_images', 'gen_%s.png' % (str(current_step).zfill(5)))
            write_images(images, filename)
            tf.logging.info('Finished generating images')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # Optimization hyperparams:
    parser.add_argument("--mode", type=str, default='train',
                        help="Mode is either train or eval")
    parser.add_argument("--train_steps", type=int, default=5000000,
                        help="Train epoch size")
    parser.add_argument("--train_steps_per_eval", type=int, default=5000 if USE_TPU else 2000,
                        help="Steps per eval and image generation")
    parser.add_argument("--iterations_per_loop", type=int, default=500 if USE_TPU else 200,
                        help="Steps per interior TPU loop")
    parser.add_argument("--num_eval_images", type=int, default=100,
                        help="Number of images for evaluation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Minibatch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=.999, help="beta2")
    parser.add_argument("--adam_eps", type=float, default=10e-5, help="eps")
    parser.add_argument("--report_histograms", type=bool, default=False,
                        help="If should report histograms")
    parser.add_argument("--memory_saving_gradients", type=bool, default=False,
                        help="Use memory saving gradients")
    parser.add_argument("--use_gradient_clipping", type=bool, default=False,
                        help="Use gradient clipping")
    parser.add_argument("--use_l2_regularization", type=bool, default=False,
                        help="Use L2 loss regularization on trainable variables")
    parser.add_argument("--l2_regularization_factor", type=float,
                        default=0.00005, help="L2 regularization factor")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=-1,
                        help="Width of hidden layers (-1 for width_dict)")
    parser.add_argument("--depth", type=int, default=-1,
                        help="Depth of network (-1 for depth_dict)")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=16,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=5,
                        help="Number of levels")
    parser.add_argument("--n_y", type=int, default=1,
                        help="Number of final layer output")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")

    # Cloud TPU Cluster Resolvers
    parser.add_argument("--use_tpu", type=bool, default=True if USE_TPU else False,
                        help="Use TPU for training")
    parser.add_argument("--num_shards", type=int,
                        default=8, help="Number of TPU shards")
    parser.add_argument("--tpu", type=str, default='$TPU_NAME' if USE_TPU else None,
                        help="The Cloud TPU to use for training")
    parser.add_argument("--gcp_project", type=str, default=None,
                        help="Project name for the Cloud TPU-enabled project")
    parser.add_argument("--tpu_zone", type=str, default=None,
                        help="GCE zone where the Cloud TPU is located in")

    # dataset
    parser.add_argument("--data_dir", type=str, default='gs://BUCKET/dataset' if USE_TPU else './dataset',
                        help="Bucket/Folder that contains the data tfrecord files")
    parser.add_argument("--model_dir", type=str, default='gs://BUCKET/output' if USE_TPU else './output',
                        help="Output model directory")

    cfg = parser.parse_args()
    cfg.width_dict = {1: 512, 2: 512, 4: 512,
                      8: 256, 16: 256, 32: 256, 64: 128, 128: 64}
    cfg.depth_dict = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}
    #cfg.depth_dict = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

    tf.logging.set_verbosity(tf.logging.INFO)
    main(cfg)
