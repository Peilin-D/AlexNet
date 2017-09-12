from model import load_pretrained_weights, inference_deep, loss, train
from eval import evaluate
from inputs import *

import tensorflow as tf
from datetime import datetime
import time
import argparse

def run_train(train_from_scratch, batch_size, use_keypts):
  with tf.Graph().as_default() as g:
    global_step = tf.contrib.framework.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images, labels, keypts = distorted_inputs(batch_size)
    if use_keypts:
      pred = inference_deep(images, 0.5, keypts)
    else:
      pred = inference_deep(images, 0.5)

    skip_layers = ['fc7', 'fc8']
    load_op = load_pretrained_weights(skip_layers)
    total_loss = loss(pred, labels, 0.000)
    train_op = train(total_loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        if self._step % 600 == 0 and self._step > 0:
          evaluate(False, 500, False)
        if self._step % 100 == 0:
          return tf.train.SessionRunArgs(total_loss)
        else:
          return None

      def after_run(self, run_context, run_values):
        if self._step % 100 == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time
          loss_val = run_values.results
          format_str = '%s: step %d, loss = %.3f, duration = %.2f'
          print format_str % (datetime.now(), self._step, loss_val, duration)

    with tf.train.MonitoredTrainingSession(
      checkpoint_dir='./tmp/ckpt_3',
      save_checkpoint_secs=300,
      hooks=[tf.train.StopAtStepHook(last_step=10000),
            tf.train.NanTensorHook(total_loss),
            _LoggerHook()]
    ) as sess:
        if not train_from_scratch:
          sess.run(load_op)
        while not sess.should_stop():
          sess.run(train_op)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from_scratch', action='store_true', default=False)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--use_keypts', action='store_true', default=False)
  args = parser.parse_args()
  run_train(args.from_scratch, args.batch_size, args.use_keypts)
