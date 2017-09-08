from featureExtractorModule import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.slim.preprocessing import vgg_preprocessing
from imageReader import *
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes


# ----------------need testing--------------------------
def sample_feature_vectors(features, label, k, instances):

    samples_indices = get_object_samples(label, k, instances)
    sampled_vectors = []

    for sampleIdx in samples_indices:
        feature_vectors = tf.gather_nd(features, sampleIdx)
        sampled_vectors.append(feature_vectors)

    if len(sampled_vectors)>1:
        sampled_vectors = tf.concat(sampled_vectors, axis=0)
    else:
        sampled_vectors = tf.convert_to_tensor(sampled_vectors)

    return sampled_vectors


def batch_loss(features_batch, labels_batch, k, batch_size):

    loss = tf.Variable(0, dtype=tf.float32)
    print("batch_loss running")
    for i in range(batch_size):
            loss += pairwise_loss(features_batch[i], labels_batch[i], k)

    return loss


# ----------------need testing--------------------------
def pairwise_loss(features, label, k):

    instances, counts = get_object_sizes(label)
    num_instances = tf.size(instances)
    mask = []

    vectors = sample_feature_vectors(features, label, k, instances)
    squared_norms = tf.reduce_sum(vectors*vectors, axis=1)
    squared_norms = tf.reshape(squared_norms, [-1,1])

    dist_matrix = squared_norms - 2*tf.matmul(vectors,vectors,transpose_b=True) + tf.transpose(squared_norms)
    dist_matrix = tf.div(2,(1+tf.exp(dist_matrix)))

    match_loss = tf.log(dist_matrix)
    mismatch_loss = tf.log(1-dist_matrix)

    match_mask, mismatch_mask, weights = get_masks_and_weights(k, num_instances, counts)

    match_loss = tf.multiply(match_loss, match_mask)
    mismatch_loss = tf.multiply(mismatch_loss, mismatch_mask)

    total_loss = tf.multiply(match_loss, weights) + tf.multiply(mismatch_loss, weights)
    total_loss = tf.reduce_sum(total_loss)

    return total_loss


def get_masks_and_weights(k, num_instances, counts):

    n = k*num_instances
    At, A = np.mgrid[0:n, 0:n]

    ones = np.ones((n,n))
    zeroes = np.zeros((n,n))

    mask = np.zeros((n,n))
    weights = np.zeros((n,n))


    for i in range(num_instances):
        M_1 = np.where(((A>=k*i) & (A<k*(i+1))), ones, zeroes)
        M_2 = np.where(((At>=k*i) & (At<k*(i+1))), ones, zeroes)
        weights = weights + M_1*counts[i] + M_2*counts[i]
        mask = np.add(mask,  np.multiply(M_1,M_2))

    weights = 1.0/weights
    weights = weights/np.sum(weights)

    return mask, 1-mask, weights


def get_object_sizes(label):

    label = tf.expand_dims(label, axis=0)
    flat_label = tf.contrib.layers.flatten(label)
    flat_label = tf.squeeze(flat_label)
    y, _, counts = tf.unique_with_counts(flat_label)

    return y, counts


def get_object_samples(label, k, instances):

    sample_locations = []
    unstacked_instances = tf.TensorArray.split(instances)
    for inst in unstacked_instances:
        if inst not in [0, 220]:
            object_indices = tf.transpose(tf.where(label==inst))
            print(object_indices)
            sample_indices = np.random.choice(object_indices[0].size, k)

            try:
                x = object_indices[0][sample_indices]
                y = object_indices[1][sample_indices]
                sample_locations.append(np.array([x, y]).T)
                break
            except IndexError:
                print("index error caught")



    return tf.convert_to_tensor(np.array(sample_locations))

