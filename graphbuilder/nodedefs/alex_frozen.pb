
I
inputPlaceholder*
dtype0*&
shape:�����������
�
9alexnet_v2/conv1/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"         @   *+
_class!
loc:@alexnet_v2/conv1/weights
�
7alexnet_v2/conv1/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *��޼*+
_class!
loc:@alexnet_v2/conv1/weights
�
7alexnet_v2/conv1/weights/Initializer/random_uniform/maxConst*
valueB
 *���<*+
_class!
loc:@alexnet_v2/conv1/weights*
dtype0
�
Aalexnet_v2/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform9alexnet_v2/conv1/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@alexnet_v2/conv1/weights*
dtype0*
seed2 *

seed 
�
7alexnet_v2/conv1/weights/Initializer/random_uniform/subSub7alexnet_v2/conv1/weights/Initializer/random_uniform/max7alexnet_v2/conv1/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
7alexnet_v2/conv1/weights/Initializer/random_uniform/mulMulAalexnet_v2/conv1/weights/Initializer/random_uniform/RandomUniform7alexnet_v2/conv1/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
3alexnet_v2/conv1/weights/Initializer/random_uniformAdd7alexnet_v2/conv1/weights/Initializer/random_uniform/mul7alexnet_v2/conv1/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
alexnet_v2/conv1/weights
VariableV2*
dtype0*
	container *
shape:@*
shared_name *+
_class!
loc:@alexnet_v2/conv1/weights
�
alexnet_v2/conv1/weights/AssignAssignalexnet_v2/conv1/weights3alexnet_v2/conv1/weights/Initializer/random_uniform*
T0*+
_class!
loc:@alexnet_v2/conv1/weights*
validate_shape(*
use_locking(
y
alexnet_v2/conv1/weights/readIdentityalexnet_v2/conv1/weights*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
8alexnet_v2/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*+
_class!
loc:@alexnet_v2/conv1/weights*
dtype0
�
9alexnet_v2/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/conv1/weights/read*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
2alexnet_v2/conv1/kernel/Regularizer/l2_regularizerMul8alexnet_v2/conv1/kernel/Regularizer/l2_regularizer/scale9alexnet_v2/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*+
_class!
loc:@alexnet_v2/conv1/weights
�
)alexnet_v2/conv1/biases/Initializer/ConstConst*
dtype0*
valueB@*���=**
_class 
loc:@alexnet_v2/conv1/biases
�
alexnet_v2/conv1/biases
VariableV2*
dtype0*
	container *
shape:@*
shared_name **
_class 
loc:@alexnet_v2/conv1/biases
�
alexnet_v2/conv1/biases/AssignAssignalexnet_v2/conv1/biases)alexnet_v2/conv1/biases/Initializer/Const*
use_locking(*
T0**
_class 
loc:@alexnet_v2/conv1/biases*
validate_shape(
v
alexnet_v2/conv1/biases/readIdentityalexnet_v2/conv1/biases*
T0**
_class 
loc:@alexnet_v2/conv1/biases
S
alexnet_v2/conv1/dilation_rateConst*
valueB"      *
dtype0
�
alexnet_v2/conv1/Conv2DConv2Dinputalexnet_v2/conv1/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

z
alexnet_v2/conv1/BiasAddBiasAddalexnet_v2/conv1/Conv2Dalexnet_v2/conv1/biases/read*
T0*
data_formatNHWC
@
alexnet_v2/conv1/ReluRelualexnet_v2/conv1/BiasAdd*
T0
�
alexnet_v2/pool1/MaxPoolMaxPoolalexnet_v2/conv1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*
T0
�
9alexnet_v2/conv2/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      @   �   *+
_class!
loc:@alexnet_v2/conv2/weights
�
7alexnet_v2/conv2/weights/Initializer/random_uniform/minConst*
valueB
 *����*+
_class!
loc:@alexnet_v2/conv2/weights*
dtype0
�
7alexnet_v2/conv2/weights/Initializer/random_uniform/maxConst*
valueB
 *���<*+
_class!
loc:@alexnet_v2/conv2/weights*
dtype0
�
Aalexnet_v2/conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform9alexnet_v2/conv2/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
7alexnet_v2/conv2/weights/Initializer/random_uniform/subSub7alexnet_v2/conv2/weights/Initializer/random_uniform/max7alexnet_v2/conv2/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
7alexnet_v2/conv2/weights/Initializer/random_uniform/mulMulAalexnet_v2/conv2/weights/Initializer/random_uniform/RandomUniform7alexnet_v2/conv2/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
3alexnet_v2/conv2/weights/Initializer/random_uniformAdd7alexnet_v2/conv2/weights/Initializer/random_uniform/mul7alexnet_v2/conv2/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
alexnet_v2/conv2/weights
VariableV2*+
_class!
loc:@alexnet_v2/conv2/weights*
dtype0*
	container *
shape:@�*
shared_name 
�
alexnet_v2/conv2/weights/AssignAssignalexnet_v2/conv2/weights3alexnet_v2/conv2/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
y
alexnet_v2/conv2/weights/readIdentityalexnet_v2/conv2/weights*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
8alexnet_v2/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*+
_class!
loc:@alexnet_v2/conv2/weights*
dtype0
�
9alexnet_v2/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/conv2/weights/read*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
2alexnet_v2/conv2/kernel/Regularizer/l2_regularizerMul8alexnet_v2/conv2/kernel/Regularizer/l2_regularizer/scale9alexnet_v2/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*+
_class!
loc:@alexnet_v2/conv2/weights
�
)alexnet_v2/conv2/biases/Initializer/ConstConst*
valueB�*���=**
_class 
loc:@alexnet_v2/conv2/biases*
dtype0
�
alexnet_v2/conv2/biases
VariableV2*
shape:�*
shared_name **
_class 
loc:@alexnet_v2/conv2/biases*
dtype0*
	container 
�
alexnet_v2/conv2/biases/AssignAssignalexnet_v2/conv2/biases)alexnet_v2/conv2/biases/Initializer/Const*
T0**
_class 
loc:@alexnet_v2/conv2/biases*
validate_shape(*
use_locking(
v
alexnet_v2/conv2/biases/readIdentityalexnet_v2/conv2/biases*
T0**
_class 
loc:@alexnet_v2/conv2/biases
S
alexnet_v2/conv2/dilation_rateConst*
valueB"      *
dtype0
�
alexnet_v2/conv2/Conv2DConv2Dalexnet_v2/pool1/MaxPoolalexnet_v2/conv2/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
z
alexnet_v2/conv2/BiasAddBiasAddalexnet_v2/conv2/Conv2Dalexnet_v2/conv2/biases/read*
T0*
data_formatNHWC
@
alexnet_v2/conv2/ReluRelualexnet_v2/conv2/BiasAdd*
T0
�
alexnet_v2/pool2/MaxPoolMaxPoolalexnet_v2/conv2/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

�
9alexnet_v2/conv3/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      �   �  *+
_class!
loc:@alexnet_v2/conv3/weights
�
7alexnet_v2/conv3/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *HY�*+
_class!
loc:@alexnet_v2/conv3/weights
�
7alexnet_v2/conv3/weights/Initializer/random_uniform/maxConst*
valueB
 *HY=*+
_class!
loc:@alexnet_v2/conv3/weights*
dtype0
�
Aalexnet_v2/conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform9alexnet_v2/conv3/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
7alexnet_v2/conv3/weights/Initializer/random_uniform/subSub7alexnet_v2/conv3/weights/Initializer/random_uniform/max7alexnet_v2/conv3/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
7alexnet_v2/conv3/weights/Initializer/random_uniform/mulMulAalexnet_v2/conv3/weights/Initializer/random_uniform/RandomUniform7alexnet_v2/conv3/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
3alexnet_v2/conv3/weights/Initializer/random_uniformAdd7alexnet_v2/conv3/weights/Initializer/random_uniform/mul7alexnet_v2/conv3/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
alexnet_v2/conv3/weights
VariableV2*+
_class!
loc:@alexnet_v2/conv3/weights*
dtype0*
	container *
shape:��*
shared_name 
�
alexnet_v2/conv3/weights/AssignAssignalexnet_v2/conv3/weights3alexnet_v2/conv3/weights/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@alexnet_v2/conv3/weights*
validate_shape(
y
alexnet_v2/conv3/weights/readIdentityalexnet_v2/conv3/weights*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
8alexnet_v2/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*+
_class!
loc:@alexnet_v2/conv3/weights*
dtype0
�
9alexnet_v2/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/conv3/weights/read*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
2alexnet_v2/conv3/kernel/Regularizer/l2_regularizerMul8alexnet_v2/conv3/kernel/Regularizer/l2_regularizer/scale9alexnet_v2/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*+
_class!
loc:@alexnet_v2/conv3/weights
�
)alexnet_v2/conv3/biases/Initializer/ConstConst*
valueB�*���=**
_class 
loc:@alexnet_v2/conv3/biases*
dtype0
�
alexnet_v2/conv3/biases
VariableV2*
shared_name **
_class 
loc:@alexnet_v2/conv3/biases*
dtype0*
	container *
shape:�
�
alexnet_v2/conv3/biases/AssignAssignalexnet_v2/conv3/biases)alexnet_v2/conv3/biases/Initializer/Const*
T0**
_class 
loc:@alexnet_v2/conv3/biases*
validate_shape(*
use_locking(
v
alexnet_v2/conv3/biases/readIdentityalexnet_v2/conv3/biases*
T0**
_class 
loc:@alexnet_v2/conv3/biases
S
alexnet_v2/conv3/dilation_rateConst*
dtype0*
valueB"      
�
alexnet_v2/conv3/Conv2DConv2Dalexnet_v2/pool2/MaxPoolalexnet_v2/conv3/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
z
alexnet_v2/conv3/BiasAddBiasAddalexnet_v2/conv3/Conv2Dalexnet_v2/conv3/biases/read*
T0*
data_formatNHWC
@
alexnet_v2/conv3/ReluRelualexnet_v2/conv3/BiasAdd*
T0
�
9alexnet_v2/conv4/weights/Initializer/random_uniform/shapeConst*%
valueB"      �  �  *+
_class!
loc:@alexnet_v2/conv4/weights*
dtype0
�
7alexnet_v2/conv4/weights/Initializer/random_uniform/minConst*
valueB
 *�[�*+
_class!
loc:@alexnet_v2/conv4/weights*
dtype0
�
7alexnet_v2/conv4/weights/Initializer/random_uniform/maxConst*
valueB
 *�[�<*+
_class!
loc:@alexnet_v2/conv4/weights*
dtype0
�
Aalexnet_v2/conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform9alexnet_v2/conv4/weights/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@alexnet_v2/conv4/weights*
dtype0*
seed2 
�
7alexnet_v2/conv4/weights/Initializer/random_uniform/subSub7alexnet_v2/conv4/weights/Initializer/random_uniform/max7alexnet_v2/conv4/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
7alexnet_v2/conv4/weights/Initializer/random_uniform/mulMulAalexnet_v2/conv4/weights/Initializer/random_uniform/RandomUniform7alexnet_v2/conv4/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
3alexnet_v2/conv4/weights/Initializer/random_uniformAdd7alexnet_v2/conv4/weights/Initializer/random_uniform/mul7alexnet_v2/conv4/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
alexnet_v2/conv4/weights
VariableV2*
dtype0*
	container *
shape:��*
shared_name *+
_class!
loc:@alexnet_v2/conv4/weights
�
alexnet_v2/conv4/weights/AssignAssignalexnet_v2/conv4/weights3alexnet_v2/conv4/weights/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@alexnet_v2/conv4/weights*
validate_shape(
y
alexnet_v2/conv4/weights/readIdentityalexnet_v2/conv4/weights*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
8alexnet_v2/conv4/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*+
_class!
loc:@alexnet_v2/conv4/weights*
dtype0
�
9alexnet_v2/conv4/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/conv4/weights/read*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
2alexnet_v2/conv4/kernel/Regularizer/l2_regularizerMul8alexnet_v2/conv4/kernel/Regularizer/l2_regularizer/scale9alexnet_v2/conv4/kernel/Regularizer/l2_regularizer/L2Loss*
T0*+
_class!
loc:@alexnet_v2/conv4/weights
�
)alexnet_v2/conv4/biases/Initializer/ConstConst*
valueB�*���=**
_class 
loc:@alexnet_v2/conv4/biases*
dtype0
�
alexnet_v2/conv4/biases
VariableV2*
dtype0*
	container *
shape:�*
shared_name **
_class 
loc:@alexnet_v2/conv4/biases
�
alexnet_v2/conv4/biases/AssignAssignalexnet_v2/conv4/biases)alexnet_v2/conv4/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0**
_class 
loc:@alexnet_v2/conv4/biases
v
alexnet_v2/conv4/biases/readIdentityalexnet_v2/conv4/biases*
T0**
_class 
loc:@alexnet_v2/conv4/biases
S
alexnet_v2/conv4/dilation_rateConst*
dtype0*
valueB"      
�
alexnet_v2/conv4/Conv2DConv2Dalexnet_v2/conv3/Relualexnet_v2/conv4/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
z
alexnet_v2/conv4/BiasAddBiasAddalexnet_v2/conv4/Conv2Dalexnet_v2/conv4/biases/read*
T0*
data_formatNHWC
@
alexnet_v2/conv4/ReluRelualexnet_v2/conv4/BiasAdd*
T0
�
9alexnet_v2/conv5/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      �     *+
_class!
loc:@alexnet_v2/conv5/weights
�
7alexnet_v2/conv5/weights/Initializer/random_uniform/minConst*
valueB
 *�2�*+
_class!
loc:@alexnet_v2/conv5/weights*
dtype0
�
7alexnet_v2/conv5/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *�2=*+
_class!
loc:@alexnet_v2/conv5/weights
�
Aalexnet_v2/conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform9alexnet_v2/conv5/weights/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@alexnet_v2/conv5/weights*
dtype0*
seed2 
�
7alexnet_v2/conv5/weights/Initializer/random_uniform/subSub7alexnet_v2/conv5/weights/Initializer/random_uniform/max7alexnet_v2/conv5/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
7alexnet_v2/conv5/weights/Initializer/random_uniform/mulMulAalexnet_v2/conv5/weights/Initializer/random_uniform/RandomUniform7alexnet_v2/conv5/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
3alexnet_v2/conv5/weights/Initializer/random_uniformAdd7alexnet_v2/conv5/weights/Initializer/random_uniform/mul7alexnet_v2/conv5/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
alexnet_v2/conv5/weights
VariableV2*
shape:��*
shared_name *+
_class!
loc:@alexnet_v2/conv5/weights*
dtype0*
	container 
�
alexnet_v2/conv5/weights/AssignAssignalexnet_v2/conv5/weights3alexnet_v2/conv5/weights/Initializer/random_uniform*
T0*+
_class!
loc:@alexnet_v2/conv5/weights*
validate_shape(*
use_locking(
y
alexnet_v2/conv5/weights/readIdentityalexnet_v2/conv5/weights*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
8alexnet_v2/conv5/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*+
_class!
loc:@alexnet_v2/conv5/weights*
dtype0
�
9alexnet_v2/conv5/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/conv5/weights/read*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
2alexnet_v2/conv5/kernel/Regularizer/l2_regularizerMul8alexnet_v2/conv5/kernel/Regularizer/l2_regularizer/scale9alexnet_v2/conv5/kernel/Regularizer/l2_regularizer/L2Loss*
T0*+
_class!
loc:@alexnet_v2/conv5/weights
�
)alexnet_v2/conv5/biases/Initializer/ConstConst*
valueB�*���=**
_class 
loc:@alexnet_v2/conv5/biases*
dtype0
�
alexnet_v2/conv5/biases
VariableV2*
shape:�*
shared_name **
_class 
loc:@alexnet_v2/conv5/biases*
dtype0*
	container 
�
alexnet_v2/conv5/biases/AssignAssignalexnet_v2/conv5/biases)alexnet_v2/conv5/biases/Initializer/Const*
use_locking(*
T0**
_class 
loc:@alexnet_v2/conv5/biases*
validate_shape(
v
alexnet_v2/conv5/biases/readIdentityalexnet_v2/conv5/biases*
T0**
_class 
loc:@alexnet_v2/conv5/biases
S
alexnet_v2/conv5/dilation_rateConst*
valueB"      *
dtype0
�
alexnet_v2/conv5/Conv2DConv2Dalexnet_v2/conv4/Relualexnet_v2/conv5/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
z
alexnet_v2/conv5/BiasAddBiasAddalexnet_v2/conv5/Conv2Dalexnet_v2/conv5/biases/read*
T0*
data_formatNHWC
@
alexnet_v2/conv5/ReluRelualexnet_v2/conv5/BiasAdd*
T0
�
alexnet_v2/pool5/MaxPoolMaxPoolalexnet_v2/conv5/Relu*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

�
9alexnet_v2/fc6/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *)
_class
loc:@alexnet_v2/fc6/weights*
dtype0
�
8alexnet_v2/fc6/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@alexnet_v2/fc6/weights
�
:alexnet_v2/fc6/weights/Initializer/truncated_normal/stddevConst*
valueB
 *
ף;*)
_class
loc:@alexnet_v2/fc6/weights*
dtype0
�
Calexnet_v2/fc6/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9alexnet_v2/fc6/weights/Initializer/truncated_normal/shape*
T0*)
_class
loc:@alexnet_v2/fc6/weights*
dtype0*
seed2 *

seed 
�
7alexnet_v2/fc6/weights/Initializer/truncated_normal/mulMulCalexnet_v2/fc6/weights/Initializer/truncated_normal/TruncatedNormal:alexnet_v2/fc6/weights/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@alexnet_v2/fc6/weights
�
3alexnet_v2/fc6/weights/Initializer/truncated_normalAdd7alexnet_v2/fc6/weights/Initializer/truncated_normal/mul8alexnet_v2/fc6/weights/Initializer/truncated_normal/mean*
T0*)
_class
loc:@alexnet_v2/fc6/weights
�
alexnet_v2/fc6/weights
VariableV2*
shape:�� *
shared_name *)
_class
loc:@alexnet_v2/fc6/weights*
dtype0*
	container 
�
alexnet_v2/fc6/weights/AssignAssignalexnet_v2/fc6/weights3alexnet_v2/fc6/weights/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@alexnet_v2/fc6/weights*
validate_shape(
s
alexnet_v2/fc6/weights/readIdentityalexnet_v2/fc6/weights*
T0*)
_class
loc:@alexnet_v2/fc6/weights
�
6alexnet_v2/fc6/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*)
_class
loc:@alexnet_v2/fc6/weights*
dtype0
�
7alexnet_v2/fc6/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/fc6/weights/read*
T0*)
_class
loc:@alexnet_v2/fc6/weights
�
0alexnet_v2/fc6/kernel/Regularizer/l2_regularizerMul6alexnet_v2/fc6/kernel/Regularizer/l2_regularizer/scale7alexnet_v2/fc6/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@alexnet_v2/fc6/weights
�
'alexnet_v2/fc6/biases/Initializer/ConstConst*
valueB� *���=*(
_class
loc:@alexnet_v2/fc6/biases*
dtype0
�
alexnet_v2/fc6/biases
VariableV2*
shape:� *
shared_name *(
_class
loc:@alexnet_v2/fc6/biases*
dtype0*
	container 
�
alexnet_v2/fc6/biases/AssignAssignalexnet_v2/fc6/biases'alexnet_v2/fc6/biases/Initializer/Const*
T0*(
_class
loc:@alexnet_v2/fc6/biases*
validate_shape(*
use_locking(
p
alexnet_v2/fc6/biases/readIdentityalexnet_v2/fc6/biases*
T0*(
_class
loc:@alexnet_v2/fc6/biases
Q
alexnet_v2/fc6/dilation_rateConst*
valueB"      *
dtype0
�
alexnet_v2/fc6/Conv2DConv2Dalexnet_v2/pool5/MaxPoolalexnet_v2/fc6/weights/read*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
t
alexnet_v2/fc6/BiasAddBiasAddalexnet_v2/fc6/Conv2Dalexnet_v2/fc6/biases/read*
T0*
data_formatNHWC
<
alexnet_v2/fc6/ReluRelualexnet_v2/fc6/BiasAdd*
T0
F
alexnet_v2/dropout6/IdentityIdentityalexnet_v2/fc6/Relu*
T0
�
9alexnet_v2/fc7/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *)
_class
loc:@alexnet_v2/fc7/weights*
dtype0
�
8alexnet_v2/fc7/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@alexnet_v2/fc7/weights*
dtype0
�
:alexnet_v2/fc7/weights/Initializer/truncated_normal/stddevConst*
valueB
 *
ף;*)
_class
loc:@alexnet_v2/fc7/weights*
dtype0
�
Calexnet_v2/fc7/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9alexnet_v2/fc7/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
7alexnet_v2/fc7/weights/Initializer/truncated_normal/mulMulCalexnet_v2/fc7/weights/Initializer/truncated_normal/TruncatedNormal:alexnet_v2/fc7/weights/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
3alexnet_v2/fc7/weights/Initializer/truncated_normalAdd7alexnet_v2/fc7/weights/Initializer/truncated_normal/mul8alexnet_v2/fc7/weights/Initializer/truncated_normal/mean*
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
alexnet_v2/fc7/weights
VariableV2*
dtype0*
	container *
shape:� � *
shared_name *)
_class
loc:@alexnet_v2/fc7/weights
�
alexnet_v2/fc7/weights/AssignAssignalexnet_v2/fc7/weights3alexnet_v2/fc7/weights/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*)
_class
loc:@alexnet_v2/fc7/weights
s
alexnet_v2/fc7/weights/readIdentityalexnet_v2/fc7/weights*
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
6alexnet_v2/fc7/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*)
_class
loc:@alexnet_v2/fc7/weights
�
7alexnet_v2/fc7/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/fc7/weights/read*
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
0alexnet_v2/fc7/kernel/Regularizer/l2_regularizerMul6alexnet_v2/fc7/kernel/Regularizer/l2_regularizer/scale7alexnet_v2/fc7/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@alexnet_v2/fc7/weights
�
'alexnet_v2/fc7/biases/Initializer/ConstConst*
valueB� *���=*(
_class
loc:@alexnet_v2/fc7/biases*
dtype0
�
alexnet_v2/fc7/biases
VariableV2*
dtype0*
	container *
shape:� *
shared_name *(
_class
loc:@alexnet_v2/fc7/biases
�
alexnet_v2/fc7/biases/AssignAssignalexnet_v2/fc7/biases'alexnet_v2/fc7/biases/Initializer/Const*
use_locking(*
T0*(
_class
loc:@alexnet_v2/fc7/biases*
validate_shape(
p
alexnet_v2/fc7/biases/readIdentityalexnet_v2/fc7/biases*
T0*(
_class
loc:@alexnet_v2/fc7/biases
Q
alexnet_v2/fc7/dilation_rateConst*
dtype0*
valueB"      
�
alexnet_v2/fc7/Conv2DConv2Dalexnet_v2/dropout6/Identityalexnet_v2/fc7/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
t
alexnet_v2/fc7/BiasAddBiasAddalexnet_v2/fc7/Conv2Dalexnet_v2/fc7/biases/read*
T0*
data_formatNHWC
<
alexnet_v2/fc7/ReluRelualexnet_v2/fc7/BiasAdd*
T0
]
(alexnet_v2/global_pool/reduction_indicesConst*
valueB"      *
dtype0
�
alexnet_v2/global_poolMeanalexnet_v2/fc7/Relu(alexnet_v2/global_pool/reduction_indices*

Tidx0*
	keep_dims(*
T0
I
alexnet_v2/dropout7/IdentityIdentityalexnet_v2/global_pool*
T0
�
9alexnet_v2/fc8/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *)
_class
loc:@alexnet_v2/fc8/weights*
dtype0
�
8alexnet_v2/fc8/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@alexnet_v2/fc8/weights*
dtype0
�
:alexnet_v2/fc8/weights/Initializer/truncated_normal/stddevConst*
valueB
 *
ף;*)
_class
loc:@alexnet_v2/fc8/weights*
dtype0
�
Calexnet_v2/fc8/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9alexnet_v2/fc8/weights/Initializer/truncated_normal/shape*
T0*)
_class
loc:@alexnet_v2/fc8/weights*
dtype0*
seed2 *

seed 
�
7alexnet_v2/fc8/weights/Initializer/truncated_normal/mulMulCalexnet_v2/fc8/weights/Initializer/truncated_normal/TruncatedNormal:alexnet_v2/fc8/weights/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@alexnet_v2/fc8/weights
�
3alexnet_v2/fc8/weights/Initializer/truncated_normalAdd7alexnet_v2/fc8/weights/Initializer/truncated_normal/mul8alexnet_v2/fc8/weights/Initializer/truncated_normal/mean*
T0*)
_class
loc:@alexnet_v2/fc8/weights
�
alexnet_v2/fc8/weights
VariableV2*
shape:� *
shared_name *)
_class
loc:@alexnet_v2/fc8/weights*
dtype0*
	container 
�
alexnet_v2/fc8/weights/AssignAssignalexnet_v2/fc8/weights3alexnet_v2/fc8/weights/Initializer/truncated_normal*
T0*)
_class
loc:@alexnet_v2/fc8/weights*
validate_shape(*
use_locking(
s
alexnet_v2/fc8/weights/readIdentityalexnet_v2/fc8/weights*
T0*)
_class
loc:@alexnet_v2/fc8/weights
�
6alexnet_v2/fc8/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*)
_class
loc:@alexnet_v2/fc8/weights*
dtype0
�
7alexnet_v2/fc8/kernel/Regularizer/l2_regularizer/L2LossL2Lossalexnet_v2/fc8/weights/read*
T0*)
_class
loc:@alexnet_v2/fc8/weights
�
0alexnet_v2/fc8/kernel/Regularizer/l2_regularizerMul6alexnet_v2/fc8/kernel/Regularizer/l2_regularizer/scale7alexnet_v2/fc8/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@alexnet_v2/fc8/weights
�
'alexnet_v2/fc8/biases/Initializer/zerosConst*
dtype0*
valueB*    *(
_class
loc:@alexnet_v2/fc8/biases
�
alexnet_v2/fc8/biases
VariableV2*
shape:*
shared_name *(
_class
loc:@alexnet_v2/fc8/biases*
dtype0*
	container 
�
alexnet_v2/fc8/biases/AssignAssignalexnet_v2/fc8/biases'alexnet_v2/fc8/biases/Initializer/zeros*
T0*(
_class
loc:@alexnet_v2/fc8/biases*
validate_shape(*
use_locking(
p
alexnet_v2/fc8/biases/readIdentityalexnet_v2/fc8/biases*
T0*(
_class
loc:@alexnet_v2/fc8/biases
Q
alexnet_v2/fc8/dilation_rateConst*
valueB"      *
dtype0
�
alexnet_v2/fc8/Conv2DConv2Dalexnet_v2/dropout7/Identityalexnet_v2/fc8/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
t
alexnet_v2/fc8/BiasAddBiasAddalexnet_v2/fc8/Conv2Dalexnet_v2/fc8/biases/read*
data_formatNHWC*
T0
[
alexnet_v2/fc8/squeezedSqueezealexnet_v2/fc8/BiasAdd*
squeeze_dims
*
T0"