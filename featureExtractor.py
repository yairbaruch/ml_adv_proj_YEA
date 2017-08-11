from featureExtractorModule import *
from imageReader import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import matplotlib.pyplot as plt
from imageReader import *
import skimage.io as io
import numpy as np
#slim = tf.contrib.slim

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94


# Path to checkpoint directory, for loading pre-trained resnet_101 weights (via tf.slim)
checkpoints_dir = 'C:\Almog\MLProject/resnet_v2_101_2017_04_14/'

# Paths to images directory (data_dir), and labels (labels_dir).
data_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\JPEGImages/'
labels_dir = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\SegmentationObject/'

# Path to a file containing names of all images and labels in directories (eg 2011_000036)
data_list = 'C:\Almog\MLProject\pascal voc\VOCtrainval_11-May-2012_2\ImageSets\Segmentation/train.txt'

mean = (R_MEAN, G_MEAN, B_MEAN)
coord = tf.train.Coordinator()

# Create an ImageReader instance to read images and labels from disk and turn into batches
# more information in ImageReader.py
reader = ImageReader(data_dir, labels_dir, data_list=data_list, input_size=(256,256), random_scale=False
                     , random_mirror=False, ignore_label=0, img_mean=mean, coord=coord)

# Collect a batch of 5 [img, label] pairs, from tensorflow queue defined by reader
inputs = reader.dequeue(5)

# ---debug print----
print("inputs:")
print(inputs)

# Define network operation. network is defined in featureExtractorModule.py
features, scaled_feature_map = extract_features(inputs[0], True)

init = tf.global_variables_initializer()

# Create an init function that loads pre-trained resnet_101 weights from checkpoint_dir
init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
            slim.get_model_variables('resnet_v2_101'))

# Training session
with tf.Session() as sess:

    sess.run(init)
    init_fn(sess)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # TODO - define loss and training op to run tf.train
            inps = inputs[1].eval()
            print(inps[1, 150, :, 0])
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
