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

import celeba
global dataset
dataset = celeba

USE_TPU = False

import models

class AdvancedLearningRateScheduler():
    patientce = 64    
    best = 9999999
    lr = 0
    wait = 0
    decay = 0.5

    def setBaseLearningRate(self,lr):
        self.lr = lr

    def apply(self, metrics):        
        shouldExport = False        
        if (metrics['loss'] < self.best):
            self.best = metrics['loss']
            self.wait = 0
            shouldExport = True
        else:
            self.wait += 1
        if self.wait > self.patientce:
            self.lr *= self.decay
            self.wait = 0
        return shouldExport
    
    def learning_rate(self):
        return self.lr

ALRS = AdvancedLearningRateScheduler()

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
            'generated_images': model.sample(y, is_training=False)        
        }
        return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)    
    real_images = features['real_images']
        
    f_loss = model.f_loss(real_images, y, is_training)    

    if mode == tf.estimator.ModeKeys.TRAIN:
        #########
        # TRAIN #
        #########
        
        f_loss = tf.reduce_mean(f_loss)

        global_step = tf.train.get_global_step()
        base_lr = cfg.lr * tf.minimum(1., 1.0 / cfg.warmup * tf.cast(global_step, tf.float32))
        ALRS.setBaseLearningRate(base_lr)
        learning_rate = ALRS.learning_rate()

        if not cfg.use_tpu:
            for v in tf.trainable_variables(): 
                tf.summary.histogram(v.name.replace(':','_'),v)
            tf.summary.scalar('lr', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=cfg.beta1, epsilon=cfg.adam_eps)

        if cfg.use_tpu:
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)            

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            step = optimizer.minimize(f_loss, var_list=tf.trainable_variables())            
            
            increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
            joint_op = tf.group([step, increment_step])

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

        current_step = estimator._load_global_step_from_checkpoint_dir(cfg.model_dir)   # pylint: disable=protected-access,line-too-long
        tf.logging.info('Starting training for %d steps, current step: %d' % (cfg.train_steps, current_step))
        while current_step < cfg.train_steps:
            next_checkpoint = min(current_step + cfg.train_steps_per_eval, cfg.train_steps)
            est.train(input_fn=dataset.InputFunction(True), max_steps=next_checkpoint)
            current_step = next_checkpoint
            tf.logging.info('Finished training step %d' % current_step)

            metrics = est.evaluate(input_fn=dataset.InputFunction(False), steps=cfg.num_eval_images // cfg.batch_size)
            tf.logging.info('Finished evaluating')
            tf.logging.info(metrics)
            ALRS.apply(metrics) 

            generated_iter = local_est.predict(input_fn=y_input_fn)
            images = []            
            images = [p['generated_images'][:, :, :] for p in generated_iter]
            filename = os.path.join(cfg.model_dir, 'generated_images', 'gen_%s.png' % (str(current_step).zfill(5)))
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
    parser.add_argument("--train_steps_per_eval", type=int, default=5000 if USE_TPU else 200,
                        help="Steps per eval and image generation")
    parser.add_argument("--iterations_per_loop", type=int, default=500 if USE_TPU else 100, 
                        help="Steps per interior TPU loop")
    parser.add_argument("--num_eval_images", type=int, default=100, 
                        help="Number of images for evaluation")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Minibatch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--warmup", type=float, default=2000.0,
                        help="Warmup steps")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--adam_eps", type=float, default=10e-5, help="Adam eps")
    
    # Model hyperparams:
    parser.add_argument("--width", type=int, default=-1,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
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
    parser.add_argument("--flow_permutation", type=int, default=0,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

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
    parser.add_argument("--data_dir", type=str, default='$STORAGE_BUCKET/dataset' if USE_TPU else './dataset',
                        help="Bucket/Folder that contains the data tfrecord files")
    parser.add_argument("--model_dir", type=str, default='$STORAGE_BUCKET/output' if USE_TPU else './output',
                        help="Output model directory")

    cfg = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)  
    main(cfg)

