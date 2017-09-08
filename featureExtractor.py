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


R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

batch_size = 5
img_h = 384
img_w = 384
is_training = True
k = 25


# Path to checkpoint directory, for loading pre-trained resnet_101 weights (via tf.slim)
checkpoints_dir = 'C:\Almog\MLProject/resnet_v2_101_2017_04_14/'

# Paths to images directory (data_dir), and labels (labels_dir).
data_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\JPEGImages/'
labels_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\SegmentationObject/'

# Path to a file containing names of all images and labels in directories (eg 2011_000036)
data_list = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\ImageSets\Segmentation/train.txt'

# Create placeholder for labels
labels = tf.placeholder(tf.uint8, (batch_size, img_h, img_w, 1))

# Create an ImageReader instance to read images and labels from disk and turn into batches
# more information in ImageReader.py
mean = (R_MEAN, G_MEAN, B_MEAN)
coord = tf.train.Coordinator()
reader = ImageReader(data_dir, labels_dir, data_list=data_list, input_size=(img_h,img_w), random_scale=is_training
                     , random_mirror=is_training, ignore_label=0, img_mean=mean, coord=coord)

# Collect a batch of batch_size [img, label] pairs, from tensorflow queue defined by reader
inputs = reader.dequeue(batch_size)
image_batch = inputs[0]
label_batch = inputs[1]

# Define network operation. network is defined in featureExtractorModule.py
features, scaled_feature_map = extract_features(image_batch, is_training)

# Define loss, optimizer, train_op
loss = batch_loss(features, label_batch, k, batch_size)
print(loss)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
# Create an init function that loads pre-trained resnet_101 weights from checkpoint_dir
init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
            slim.get_model_variables('resnet_v2_101'))


# Training session
with tf.Session() as sess:

    sess.run(init_global)
    sess.run(init_local)
    init_fn(sess)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # TODO - define loss and training op to run tf.train
            #evaluated_labels = label_batch
            #sess.run(train_op, feed_dict={labels:evaluated_labels})
            sess.run(train_op)
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