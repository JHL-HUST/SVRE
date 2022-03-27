# coding=utf-8
"""Implementation of SVRE-I-FGSM Attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from posix import XATTR_CREATE
import numpy as np
import cv2
import pandas as pd
import scipy.stats as st
from imageio import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import random

slim = tf.contrib.slim
tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_integer('m_svrg', 16, 'hyper-parameter of internal update frequency')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dataset/images/',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './results/SVRE-I-FGSM/',
                       'Output directory with images.')

tf.flags.DEFINE_integer('seed', 1000, 'seed num')

FLAGS = tf.flags.FLAGS
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)
model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensemble_graph(x, one_hot, num_classes = 1001):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_inc_res, end_points_inc_res = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_res_v2, end_points_res_v2 = resnet_v2.resnet_v2_101(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    
    logits = ( logits_inc_v3 + logits_inc_v4 +logits_inc_res+ logits_res_v2) / 4
    auxlogits = (end_points_inc_v3['AuxLogits'] + end_points_inc_v4['AuxLogits'] + end_points_inc_res['AuxLogits']) / 3
    #auxlogits = ( end_points_inc_v3['AuxLogits'] + end_points_inc_v4['AuxLogits']) /2
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)
    noise = tf.gradients(cross_entropy, x)[0]
    return noise


def single_graph_inc_v3(x, one_hot, num_classes = 1001):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_v3, end_points_inc_v3['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]

    return noise


def single_graph_inc_v4(x, one_hot, num_classes = 1001):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_v4, end_points_inc_v4['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]

    return noise


def single_graph_inc_res(x, one_hot, num_classes = 1001):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_inc_res, end_points_inc_res = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_res, end_points_inc_res['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]

    return noise


def single_graph_res_v2(x, one_hot, num_classes = 1001):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_res_v2, end_points_res_v2 = resnet_v2.resnet_v2_101(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits = logits_res_v2
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)

    noise = tf.gradients(cross_entropy, x)[0]

    return noise


def main(_):
    f2l = load_labels('./dataset/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    beta = alpha
    m_svrg = FLAGS.m_svrg
    num_classes = 1001

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y_input = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)

        one_hot = tf.one_hot(y_input, num_classes)

        grad_ensemble = ensemble_graph(x_input, one_hot)
        grad_inc_v3 = single_graph_inc_v3(x_input, one_hot)
        grad_inc_v4 = single_graph_inc_v4(x_input, one_hot)
        grad_inc_res = single_graph_inc_res(x_input, one_hot)
        grad_res_v2 = single_graph_res_v2(x_input, one_hot)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['inception_v4'])
            s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s4.restore(sess, model_checkpoint_map['resnet_v2'])

            idx = 0
            l2_diff = 0
        
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                labels = []
                for filename in filenames:
                    labels.append(f2l[filename])

                x = np.copy(images)
                x_min = np.clip(x - eps, -1.0, 1.0)
                x_max = np.clip(x + eps, -1.0, 1.0)

                grad = np.zeros(shape=batch_shape, dtype=np.float32)
   
                for i in range(num_iter):
                    # compute the batch gradient
                    noise_ensemble = sess.run(grad_ensemble, feed_dict={x_input: x, y_input: labels})
                    x_inner = np.copy(x)
                    x_before = np.copy(x)
                    noise_inner_all = np.zeros([m_svrg, *x.shape])
                    grad_inner = np.zeros_like(x)
                    for j in range(m_svrg):
                        # choose model uniformly from model pool
                        grad_single = np.random.choice([grad_inc_v3, grad_inc_v4, grad_inc_res, grad_res_v2])

                        # get estimate gradient
                        noise_x = sess.run(grad_single, feed_dict={x_input: x, y_input: labels})
                        noise_x_inner = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})
                        noise_inner = noise_x_inner - (noise_x - noise_ensemble)

                        noise_inner = noise_inner / np.mean(np.abs(noise_inner), (1, 2, 3), keepdims=True)
                        grad_inner = grad_inner + noise_inner

                        # update inner adversarial example
                        x_inner = x_inner + beta * np.sign(grad_inner)
                        x_inner = np.clip(x_inner, x_min, x_max)
                        
                        noise_inner_all[j] = np.copy(grad_inner)
                    
                    noise = np.copy(noise_inner_all[-1])
                   
                    x = x_before + alpha * np.sign(noise)
                    x = np.clip(x, x_min, x_max)

                save_images(x, filenames, FLAGS.output_dir)

                diff = (x + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
