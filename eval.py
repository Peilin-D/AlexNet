import tensorflow as tf
from inputs import *
from model import inference_deep, inference
import argparse

def restoreEMA(skip_layers):
  variable_averages = tf.train.ExponentialMovingAverage(0.999)
  ema_variables = variable_averages.variables_to_restore()
  for k in ema_variables.keys():
    layer = k.split('/')[0]
    if layer in skip_layers:
      del ema_variables[k]
  
  return ema_variables

def evaluate(train, batch_size, use_keypts, checkpoint_dir):
  with tf.Graph().as_default() as g:
    if train:
      images, labels, keypts = distorted_inputs(batch_size)
    else:
      images, labels, keypts = inputs(batch_size)

    # if use_keypts:
    #  pred = inference_deep(images, 1.0, keypts)
    # else:
    pred = inference(images, 1.0)
    top_k_op = tf.nn.in_top_k(pred, labels, 1)
    saver = tf.train.Saver(restoreEMA())
#         summary_op = tf.summary.merge_all()
#         summary_write = tf.summary.FileWriter('./tmp/eval', g)
    with tf.Session() as sess:
      print checkpoint_dir
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print ckpt.all_model_checkpoint_paths
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action='store_true', default=False)
  parser.add_argument('--batch_size', default=1000, type=int)
  parser.add_argument('--use_keypts', action='store_true', default=False)
  parser.add_argument('--ckpt', required=True)
  args = parser.parse_args()
  evaluate(args.train, args.batch_size, args.use_keypts, args.ckpt)
