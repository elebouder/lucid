from freeze_graph import freeze_graph
from export_graph import export_graph
import tensorflow as tf


tf.app.flags.DEFINE_string('op', 'e', 'Either e or f, the operation to complete')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', 299,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_integer(
    'label_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string('node_def_file', '/home/elebouder/TNSRVIS/lucid/graphbuilder/nodedefs/inception_resnet_v2_frozen.pb', 'location of node def file.')

tf.app.flags.DEFINE_string(
    'netname', 'inception_resnet_v2', 'name of network')

tf.app.flags.DEFINE_string(
    'net_arg_scopename', 'inception_resnet_v2_arg_scope', 'network scope name')

tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes')

tf.app.flags.DEFINE_boolean('global_pool_exists', False, 'whether or not the network takes global pool as input')

tf.app.flags.DEFINE_string("input_saver", "",
                           """TensorFlow saver file to load.""")
tf.app.flags.DEFINE_string("input_checkpoint", "/home/elebouder/CXR_TF/tf_models/inceptionresnet_slim_v2/log/model.ckpt-308151", """TensorFlow variables file to load.""")
tf.app.flags.DEFINE_string("output_graph", "/home/elebouder/TNSRVIS/lucid/graphbuilder/graphdefs/inception_resnet_v2_frozen.pb.modelzoo","""Output 'GraphDef' file name.""")
tf.app.flags.DEFINE_boolean("input_binary", True,
                            """Whether the input files are in binary format.""")
tf.app.flags.DEFINE_string("output_node_names", "InceptionResnetV2/Logits/Predictions",
                           """The name of the output nodes, comma separated.""")
tf.app.flags.DEFINE_string("restore_op_name", "save/restore_all",
                           """The name of the master restore operator.""")
tf.app.flags.DEFINE_string("filename_tensor_name", "save/Const:0",
                           """The name of the tensor holding the save path.""")
tf.app.flags.DEFINE_boolean("clear_devices", True,
                            """Whether to remove device specifications.""")
tf.app.flags.DEFINE_string("initializer_nodes", "", "comma separated list of "
                           "initializer nodes to run before freezing.")


FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.op=='e':
        export_graph(FLAGS.netname, FLAGS.net_arg_scopename, FLAGS.input_checkpoint, outfile=FLAGS.node_def_file, label_offset=FLAGS.label_offset, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size, is_training=FLAGS.is_training, num_classes=FLAGS.num_classes, global_pool_exists=FLAGS.global_pool_exists)
    elif FLAGS.op=='f':
        freeze_graph(FLAGS.node_def_file, input_saver=FLAGS.input_saver, input_binary=FLAGS.input_binary, input_checkpoint=FLAGS.input_checkpoint, output_node_names=FLAGS.output_node_names, restore_op_name=FLAGS.restore_op_name, filename_tensor_name=FLAGS.filename_tensor_name, output_graph=FLAGS.output_graph, clear_devices=FLAGS.clear_devices, initializer_nodes=FLAGS.initializer_nodes)
    else:
        print('op Args can only be one of "e" or "f"')


if __name__ == '__main__':
    tf.app.run()
