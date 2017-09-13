from featureExtractorModule import *
from featureExtractorUtils import *
import sys
sys.path.append('C:\Almog\MLProject/featureExtractor\ml_adv_proj_YEA\models\slim')
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import matplotlib.pyplot as plt
from imageReader import *
import skimage.io as io
import numpy as np
#slim = tf.contrib.slim
from datasets import imagenet
from preprocessing import inception_preprocessing

tf.logging.set_verbosity(tf.logging.INFO)

batch_size = 3
img_h = 192
img_w = 192
is_training = True
k = 25

logs_path = 'C:\Almog\MLProject/logs'
# Path to checkpoint directory, for loading pre-trained resnet_101 weights (via tf.slim)
checkpoints_dir = 'C:\Almog\MLProject/resnet_v2_101_2017_04_14/'

# Paths to images directory (data_dir), and labels (labels_dir).
data_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\JPEGImages/'
labels_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\SegmentationObject/'

# Path to a file containing names of all images and labels in directories (eg 2011_000036)
data_list = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\ImageSets\Segmentation/train.txt'

# Create placeholders
labels = tf.placeholder(tf.uint8, shape=[batch_size, img_h, img_w, 1])

sample_indices_holders = []
masks_holders = []
weights_holders = []
for i in range(batch_size):
    sample_indices_holders.append(tf.placeholder(tf.int32, shape=[None, 2], name=str(i)+'_sample_holder'))
    masks_holders.append(tf.placeholder(tf.int32, shape=[2,None,None], name=str(i)+'_masks_holder'))
    weights_holders.append(tf.placeholder(tf.float32, shape=[None,None], name=str(i)+'_weights_holder'))
sampled_vectors_batch = tf.placeholder(tf.float32, shape=[batch_size, None, 64])

# Create an ImageReader instance to read images and labels from disk and turn into batches
# more information in ImageReader.py
coord = tf.train.Coordinator()
reader = ImageReader(data_dir, labels_dir, data_list=data_list, input_size=(img_h,img_w), random_scale=is_training
                     , random_mirror=is_training, ignore_label=0, coord=coord)

# Collect a batch of batch_size [img, label] pairs, from tensorflow queue defined by reader
inputs = reader.dequeue(batch_size)
image_batch = inputs[0]
label_batch = inputs[1]

contexts = tf.random_uniform((batch_size,1))

# Define network operation. network is defined in featureExtractorModule.py
features, scaled_feature_map = extract_features(image_batch, contexts, is_training)

# Define loss, optimizer, train_op
loss = batch_loss_eval(features, labels, sample_indices_holders, masks_holders, weights_holders, k, batch_size)
#idea for loss =====> sample vectors outside of loss, using evaluated img and label, and calculate loss based on these vectors
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
# Create an init function that loads pre-trained resnet_101 weights from checkpoint_dir
init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
            slim.get_model_variables('resnet_v2_101'))

writer = tf.summary.FileWriter(logs_path)

# Training session
with tf.Session() as sess:

    sess.run(init_global)
    sess.run(init_local)
    #init_fn(sess)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # TODO - define loss and training op to run tf.train
            evaluated_labels = np.array(sess.run(label_batch))
            sample_indices_batch, masks, weights = get_instance_samples_batch(evaluated_labels, k, batch_size)

            feed_dict = dict(zip(sample_indices_holders, sample_indices_batch))
            feed_dict.update(dict(zip(masks_holders, masks)))
            feed_dict.update(dict(zip(weights_holders, weights)))
            feed_dict[labels] = evaluated_labels

            loss_e, _ = sess.run([loss,train_op], feed_dict=feed_dict)
            print(loss_e)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()



def print_class(probs):

    probabilities = np.squeeze(probs)
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                        key=lambda x: x[1])]
    plt.imshow(np.squeeze(network_input))
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities. Pay attention that the index with
        # class names is shifted by 1 -- this is because some networks
        # were trained on 1000 classes and others on 1001. VGG-16 was trained
        # on 1000 classes.
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index]))
    plt.show()

    return None