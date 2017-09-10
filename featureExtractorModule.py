import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import skimage.io as io

slim = tf.contrib.slim

# input images dimensions. replace with values from ini file
height = 128
width = 128
'''
 Code for feature extracting network. this network takes a batch of images shaped (batch, h, w, 3) and outputs
a (batch, h, w, 64) matrix with a 64d feature vector for each pixel in an input image. size is guaranteed to perfectly 
match only if image dimensions are a power of 2 (256*256, 256*512 etc.)

 The network is based on resnet_v2_101 model imported from tf.contrib.slim, and can optionally use weights 
pre-trained on ImageNet.

 Upscaling is done using transposed convolutions (tf.contrib.layers.conv2d_transpose) and each layer includes 
batch normalization.
'''
def extract_features(inputs, contexts, is_training):

    # TODO - Add skip connections between conv-deconv layers
    with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training=is_training)):

        conv1 = tf.layers.conv2d(inputs, filters=128, kernel_size=3, strides=1, padding='same')

        context_layer = expand_context(contexts, height, width)

        contexted_batch = tf.concat([conv1, context_layer], axis=3)

        net, end_points = resnet_v2.resnet_v2_101(contexted_batch,
                                                None,
                                                global_pool=False,
                                                output_stride=16)

        deconv1 = deconv_block(net,num_inputs=2048, num_outputs=1024,
                               is_training=is_training, scope='deconv1')

        deconv2 = deconv_block(deconv1, num_inputs=1024, num_outputs=512,
                               stride=2, is_training=is_training, scope='deconv2')

        deconv3 = deconv_block(deconv2, num_inputs=512, num_outputs=256,
                               stride=2, is_training=is_training, scope='deconv3')

        deconv4 = deconv_block(deconv3, num_inputs=256,num_outputs=128,
                               stride=2, is_training=is_training, scope='deconv4')

        deconv5 = deconv_block(deconv4, num_inputs=128,num_outputs=64,
                               stride=2, is_training=is_training, scope='deconv5')

    return deconv5, net


def deconv_block(inputs, num_inputs, num_outputs, stride=1, kernel_size=3, is_training=False, scope=""):

    with tf.variable_scope(scope+'_block'):
        layer1 = tf.contrib.layers.conv2d_transpose(inputs, num_inputs, kernel_size, 1,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                trainable=is_training, scope=scope+'_1')

        output = tf.contrib.layers.conv2d_transpose(tf.add(layer1, inputs), num_outputs, kernel_size, stride,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                trainable=is_training, scope=scope + '_2')

    return output


def expand_context(input, h, w):

    tmp = tf.expand_dims(input, axis=1)
    tmp = tf.expand_dims(tmp, axis=1)
    output = tf.tile(tmp, [1,h,w,1])

    return output

