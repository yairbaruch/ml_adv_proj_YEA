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
def batch_loss_eval(features_batch, labels_batch, instance_sample_batch,  k, batch_size):

    sampled_vectors_batch = sample_feature_vectors_batch(features_batch, instance_sample_batch, batch_size)

    loss = []
    for i in range(batch_size):
            loss.append(pairwise_loss(sampled_vectors_batch[i], labels_batch[i], k))

    loss = tf.convert_to_tensor(loss)
    return tf.reduce_sum(loss)


# ----------------need testing--------------------------
def pairwise_loss(sampled_vectors, label, k):

    instances, counts = get_object_sizes(label)
    num_instances = np.size(instances)

    vectors = sampled_vectors
    squared_norms = tf.norm(vectors, axis=1)
    squared_norms = tf.square(squared_norms)

    dist_matrix = squared_norms - 2*tf.matmul(vectors,vectors,transpose_b=True) + tf.transpose(squared_norms)
    dist_matrix = tf.div(2.0,(1+tf.exp(dist_matrix)))

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


def sample_feature_vectors_batch(features_batch, instance_sample_batch, batch_size):

    sampled_vectors_batch =[]
    for i in range(batch_size):
        sampled_vectors_batch.append(sample_feature_vectors(features_batch[i], instance_sample_batch[i]))

    return tf.convert_to_tensor(sampled_vectors_batch)


def sample_feature_vectors(features, sample):

    sampled_vectors = tf.gather_nd(features, sample)

    return sampled_vectors


def get_object_sizes(label):

    label = np.expand_dims(label, axis=0)
    flat_label = label.flatten()
    flat_label = np.squeeze(flat_label)
    y, counts = np.unique(flat_label,return_counts=True)

    return y, counts

def get_instance_samples_batch(label_batch, k, batch_size):

    instance_sample_batch =[]

    for i in range(batch_size):
        instance_sample_batch.append(get_instance_samples(label_batch[i], k))

    return instance_sample_batch


def get_instance_samples(label, k):

    instances, counts = get_object_sizes(label)

    sample_locations = []
    for inst in instances:
        if inst not in [0, 220]:
            object_indices = np.transpose(np.where(label==inst))
            sample_indices = np.random.choice(object_indices[0].size, k)

            x = object_indices[0][sample_indices]
            y = object_indices[1][sample_indices]
            sample_locations.extend(np.array([x, y]).T)

    return np.array(sample_locations)
