from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
        })
    image = tf.image.decode_jpeg(features['image'])
    image = tf.cast(image, tf.float32) / 255.0 - 0.5
    image = tf.reshape(image, [3, 128*128])
    # tf.cast(features['labels'], tf.int32)
    labels = tf.constant(-1.0, shape=[40])
    return image, labels


class InputFunction(object):

    def __init__(self, is_training):
        self.is_training = is_training

    def __call__(self, params):
        batch_size = params['batch_size']
        data_dir = params['data_dir']
        file_pattern = os.path.join(data_dir, 'data_*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat()

        def fetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(
                filename, buffer_size=8 * 1024 * 1024)
            return dataset
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=8, sloppy=True))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.prefetch(8)
        dataset = dataset.map(parser, num_parallel_calls=8)
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        #dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        images, labels = dataset.make_one_shot_iterator().get_next()
        images = tf.reshape(images, [batch_size, 128, 128, 3])
        images = tf.image.random_flip_left_right(images)
        y = tf.constant(0, dtype=tf.int32, shape=[batch_size, 1])
        features = {
            'real_images': images,
            'y': y
        }
        return features, labels
