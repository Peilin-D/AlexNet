from model import load_pretrained_weights, inference, inference_deep, loss, train
from eval import evaluate
from inputs import *

import tensorflow as tf
from datetime import datetime
import time
import argparse

def run_train(train_from_scratch, batch_size, use_keypts, checkpoint_dir):
  with tf.Graph().as_default() as g:
    global_step = tf.contrib.framework.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images, labels, keypts = distorted_inputs(batch_size)
    #if use_keypts:
    #  pred = inference(images, 0.5, keypts)
    #else:
    pred = inference(images, 0.5)

    skip_layers = ['conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    load_op = load_pretrained_weights(skip_layers, set_trainable=True)
    total_loss = loss(pred, labels, 0.000)
    train_op = train(total_loss, global_step)
    saver = tf.train.Saver(max_to_keep=10)
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
      else:
        print 'No checkpoint file found'
        sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess, coord)
      if not train_from_scratch:
        sess.run(load_op)
      i = 0
      max_iter = 10000
      while i < max_iter and not coord.should_stop():
        sess.run(train_op)
        if i % 100 == 0:
          loss_val = sess.run(total_loss)
          print 'step %d, loss = %.3f' % (i, loss_val)
        if i > 0 and i % 500 == 0:
          saver.save(sess, checkpoint_dir + '/model_ckpt', global_step)
          evaluate(False, 500, False, checkpoint_dir=checkpoint_dir)
          evaluate(True, 1000, False, checkpoint_dir=checkpoint_dir)
        i += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from_scratch', action='store_true', default=False)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--use_keypts', action='store_true', default=False)
  parser.add_argument('--ckpt', required=True)
  args = parser.parse_args()
  run_train(args.from_scratch, args.batch_size, args.use_keypts, args.ckpt)
