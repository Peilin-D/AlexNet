import tensorflow as tf
from inputs import *
from model import inference_deep
import argparse

def evaluate(train, batch_size, use_keypts):
  with tf.Graph().as_default() as g:
    if train:
      images, labels, keypts = distorted_inputs(batch_size)
    else:
      images, labels, keypts = inputs(batch_size)

    if use_keypts:
      pred = inference_deep(images, 1.0, keypts)
    else:
      pred = inference_deep(images, 1.0)
    top_k_op = tf.nn.in_top_k(pred, labels, 1)
    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
#         summary_op = tf.summary.merge_all()
#         summary_write = tf.summary.FileWriter('./tmp/eval', g)
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state('./tmp/ckpt_3')
      if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print 'No checkpoint file found'
        return
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess, coord)
      try:
        true_count = np.sum(sess.run(top_k_op))
        precision = float(true_count) / batch_size
        print '%s precision: %.3f' % ('train' if train else 'test', precision)
      except Exception as e:
        print e
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

# use this function during training
def eval_once(batch_size, use_keypts, top_k): 
  images, labels, keypts = inputs(batch_size)
  if use_keypts:
    pred = inference_deep(images, 1.0, keypts)
  else:
    pred = inference_deep(images, 1.0)

  true_count = tf.reduce_sum(tf.cast(tf.nn.in_top_k(pred, labels, top_k), tf.float32))
  acc = tf.divide(true_count, batch_size)
  return acc

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action='store_true', default=False)
  parser.add_argument('--batch_size', default=1000, type=int)
  parser.add_argument('--use_keypts', action='store_true', default=False)
  args = parser.parse_args()
  evaluate(args.train, args.batch_size, args.use_keypts)
