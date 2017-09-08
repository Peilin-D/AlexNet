import numpy as np
import tensorflow as tf
import math

def conv_relu(name, input, k_size, stride, padding, groups):
  # k_size needs to be [h, w, num_in, num_out]
  input_channels = int(input.get_shape()[-1])
  assert input_channels % groups == 0
  assert k_size[3] % groups == 0
  with tf.variable_scope(name) as scope:
    k_size[2] /= groups
    kernel = tf.get_variable('weights', k_size, tf.float32, 
                             tf.truncated_normal_initializer(stddev=math.sqrt(2.0/(np.prod(k_size[:3])))),
                             tf.nn.l2_loss)
    biases = tf.get_variable('biases', [k_size[3]], tf.float32, tf.constant_initializer(0.0))
    if groups == 1:
        conv = tf.nn.conv2d(input, kernel, stride, padding=padding)
    else:
        input_groups = tf.split(input, groups, 3)
        kernel_groups = tf.split(kernel, groups, 3)
        output_groups = [tf.nn.conv2d(i, k, stride, padding=padding) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
        
    conv = tf.verify_tensor_all_finite(conv, name + ' infinite error!!!')
  return tf.nn.relu(conv + biases, name=scope.name)

def fc(name, inputs, output_size, relu=True):
    # inputs should be flattened to a 2D tensor
    input_size = int(inputs.get_shape()[1])
    with tf.variable_scope(name) as scope:
      W = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                          tf.truncated_normal_initializer(stddev=math.sqrt(2.0/input_size)),
                          tf.nn.l2_loss)
      b = tf.get_variable('biases', [output_size], tf.float32, tf.constant_initializer(0.0))
      out = tf.verify_tensor_all_finite(tf.matmul(inputs, W) + b, name + ' inifite error!!!')
      if relu:
        return tf.nn.relu(out, name=scope.name)
      else:
        return out
      
def inference_deep(images, keep_prob, keypts = None):
  # 1st layer
  conv1 = conv_relu('conv1', images, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', 1)
  lrn1 = tf.nn.local_response_normalization(conv1, alpha=2e-5, beta=0.75,
                                            depth_radius=2, bias=1.0, name='lrn1')
  pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], 
                         strides=[1, 2, 2, 1], padding='VALID', name='pool1')
  
  # 2nd layer
  conv2 = conv_relu('conv2', pool1, [5, 5, 96, 256], [1, 1, 1, 1], 'SAME', 2)
  lrn2 = tf.nn.local_response_normalization(conv2, alpha=2e-5, beta=0.75, 
                                            depth_radius=2, bias=1.0, name='lrn2')
  pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], 
                         strides=[1, 2, 2, 1], padding='VALID', name='pool2')
  
  # 3rd layer
  conv3 = conv_relu('conv3', pool2, [3, 3, 256, 384], [1, 1, 1, 1], 'SAME', 1)
      
  # 4th layer
  conv4 = conv_relu('conv4', conv3, [3, 3, 384, 384], [1, 1, 1, 1], 'SAME', 2)
      
  # 5th layer
  conv5 = conv_relu('conv5', conv4, [3, 3, 384, 256], [1, 1, 1, 1], 'SAME', 2)
  pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
  
  # 6th layer
  pool5_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
  if keypts is not None:
    pool5_flat = tf.concat([pool5_flat, keypts], axis=1)

  fc6 = fc('fc6', pool5_flat, 4096)
  
  # drop out
  fc6_drop = tf.nn.dropout(fc6, keep_prob)
  
  # 7th layer
  fc7 = fc('fc7', fc6_drop, 4096)
  
  # drop out
  fc7_drop = tf.nn.dropout(fc7, keep_prob)
  
  # readout layer
  fc8 = fc('fc8', fc7_drop, 40, relu=False)
  
  return fc8

def load_pretrained_weights(skip_layers, set_untrainable=True, warm_start=True):
  weights = np.load('bvlc_alexnet.npy').item()
  ops = []
  trainables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
  regularizations = tf.get_collection_ref(tf.GraphKeys.REGULARIZATION_LOSSES)
  for layer in weights:
    if layer not in skip_layers or warm_start:
      with tf.variable_scope(layer, reuse=True) as scope:
        kernel = tf.get_variable('weights')
        ops.append(tf.assign(kernel, weights[layer][0]))
        biases = tf.get_variable('biases')
        ops.append(tf.assign(biases, weights[layer][1]))
        if set_untrainable:
          trainables.remove(kernel)
          trainables.remove(biases)
          regularizations.remove(scope.name)
  return tf.group(*ops)

def loss(logits, labels, wd):
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return cross_entropy + wd * tf.add_n(reg_losses)

def train(loss, global_step):
  lr = tf.train.exponential_decay(0.01, global_step, 5000, 0.1, True)
  # opt = tf.train.GradientDescentOptimizer(lr)
  # opt = tf.train.MomentumOptimizer(lr, 0.9)
  opt = tf.train.AdamOptimizer()
  train_op = opt.minimize(loss, global_step)
  variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
  variable_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([train_op, variable_averages_op]):
    op = tf.no_op()
  return op

