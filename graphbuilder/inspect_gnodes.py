import tensorflow as tf

#graph_file = "./nodedefs/inception_resnet_v2_frozen.pb"
graph_file = "./nodedefs/alex_frozen.pb"
graph_def = tf.GraphDef()
with open(graph_file, "rb") as f:
  graph_def.ParseFromString(f.read())
for node in graph_def.node:
  print(node.name)
