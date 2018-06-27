"""Saves out a GraphDef containing the architecture of the model.
To use it, run something like this, with a model name defined by slim:
bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:
bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1
The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb
To run the resulting graph in C++, you can look at the label_image sample code:
bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from netdefs.alexnet_slim import alexnet_v2_arg_scope, alexnet_v2
from netdefs.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim


net_map = {
  'alexnet_v2': alexnet_v2,
  'inception_resnet_v2': inception_resnet_v2
  }


arg_scope_map = {
  'alexnet_v2_arg_scope': alexnet_v2_arg_scope,
  'inception_resnet_v2_arg_scope': inception_resnet_v2_arg_scope
  }



def export_graph(netname, net_arg_scopename, checkpoint, outfile='', label_offset=0, batch_size=None, image_size=224, is_training=False, num_classes=2, global_pool_exists=True):
  if not outfile:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    image_size = image_size
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[batch_size, image_size,
                                        image_size, 3])
    net = net_map[netname]
    net_arg = arg_scope_map[net_arg_scopename]
    if global_pool_exists:
        with slim.arg_scope(net_arg()):
            logits, end_points = net(placeholder, num_classes = 2, is_training = False, global_pool=True)
    else:
        with slim.arg_scope(net_arg()):
            logits, end_points = net(placeholder, num_classes = 2, is_training = False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        #network_fn(placeholder)
        graph_def = graph.as_graph_def()
        with gfile.GFile(outfile, 'wb') as f:
            f.write(graph_def.SerializeToString())


