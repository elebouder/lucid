# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lucid.modelzoo.vision_base import Model


def populate_inception_bottlenecks(scope):
  """Add Inception bottlenecks and their pre-Relu versions to the graph."""
  graph = tf.get_default_graph()
  for op in graph.get_operations():
    if op.name.startswith(scope+'/') and 'Concat' in op.type:
      name = op.name.split('/')[1]
      pre_relus = []
      for tower in op.inputs[1:]:
        if tower.op.type == 'Relu':
          tower = tower.op.inputs[0]
        pre_relus.append(tower)
      concat_name = scope + '/' + name + '_pre_relu'
      _ = tf.concat(pre_relus, -1, name=concat_name)


def populate_inceptionres_bottlenecks(scope):
  """Add Inception bottlenecks and their pre-Relu versions to the graph."""
  graph = tf.get_default_graph()
  for op in graph.get_operations():
    if op.name.startswith(scope+'/') and 'Concat' in op.type:
      name = op.name.split('/')[1]
      #print(name)
      pre_relus = []
      for tower in op.inputs[1:]:
        if tower.op.type == 'Relu':
          tower = tower.op.inputs[0]
          #print (tower.op.name)
          #print (tower.dtype)
        if tower.dtype == 'float32':
          pre_relus.append(tower)
      concat_name = scope + '/' + name + '_pre_relu'
      _ = tf.concat(pre_relus, -1, name=concat_name)



class InceptionV1(Model):
  model_path = 'gs://modelzoo/InceptionV1.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt'
  image_shape = [224, 224, 3]
  image_value_range = (-117, 255-117)
  input_name = 'input:0'

  def post_import(self, scope):
    populate_inception_bottlenecks(scope)


class AlexnetV2(Model):
  model_path = '/home/elebouder/TNSRVIS/lucid/graphbuilder/graphdefs/test.pb.modelzoo'
  image_shape = [224, 224, 3]
  image_value_range = (0, 1)
  input_name = 'input'


class InceptionResnet2(Model):
  model_path = '/home/elebouder/TNSRVIS/lucid/graphbuilder/graphdefs/inception_resnet_v2_frozen.pb.modelzoo'
  image_shape = [299, 299, 3]
  image_value_range = (0, 1)
  input_name = 'input'

  def post_import(self, scope):
    populate_inceptionres_bottlenecks(scope)
