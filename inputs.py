import os
import numpy as np
# from scipy import misc
import tensorflow as tf
import json
import random


def extract_train_test():
  train_map = {}
  for annotations in os.listdir('ImageSplits'):
    if annotations == 'actions.txt':
      continue
    if annotations.endswith('train.txt'):
      cls = '_'.join(annotations.split('_')[:-1])
      if cls not in train_map:
        train_map[cls] = set()
      with open('ImageSplits/' + annotations) as f:
        for line in f:
          train_map[cls].add(line.strip())
      train_folder = 'train/' + cls
      test_folder = 'test/' + cls
      if not os.path.exists(train_folder):
        os.makedirs(train_folder)
      if not os.path.exists(test_folder):
        os.makedirs(test_folder)
  for img_file in os.listdir('JPEGImages'):
      cls = '_'.join(img_file.split('_')[:-1])
      if img_file in train_map[cls]:
        os.rename('JPEGImages/' + img_file, 'train/' + cls + '/' + img_file)
      else:
        os.rename('JPEGImages/' + img_file, 'test/' + cls + '/' + img_file)

def get_keypts(folder, img_list, train):
  results = []
  i = 0
  j = 0
  if train:
    keypts_files = os.listdir('train/' + folder + '/scaled_keypts')
  else:
    keypts_files = os.listdir('test/' + folder + '/scaled_keypts')
  while i < len(img_list) and j < len(keypts_files):
    img_num = img_list[i].rstrip('.jpg').split('_')[-1]
    result = []
    if img_num in keypts_files[j]:
      if train:
        with open('train/' + folder + '/scaled_keypts/' + keypts_files[j]) as js_f:
          js = json.load(js_f)
      else:
        with open('test/' + folder + '/scaled_keypts/' + keypts_files[j]) as js_f:
          js = json.load(js_f)
      
      result = [val for pair in js for val in pair]
      i += 1
      j += 1
    else:
      result = [0.0] * (18 * 2)
      i += 1
    results.append(result)
  while i < len(img_list):
      results.append([0.0] * 36)
      i += 1
  return results

def distorted_inputs(batch_size):
  img_list = []
  label_list = []
  keypts_list = []
  i = 0
  for folder in os.listdir('train'):
    if folder == '.DS_Store':
      continue
    cur_list = ['train/' + folder + '/' + img 
                for img in os.listdir('train/' + folder) if img.endswith('.jpg')]
    label_list.extend([i] * len(cur_list))
    keypts_list.extend(get_keypts(folder, cur_list, train=True))
    img_list.extend(cur_list)
    i += 1

  input_tensor = zip(img_list, label_list, keypts_list)
  random.shuffle(input_tensor)
  
  data_queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.int32, tf.float32], shapes=[[],[],[36]])
  enqueue_op = data_queue.enqueue_many([img_list, label_list, keypts_list])
  qr = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
  tf.train.add_queue_runner(qr)
  
  img_file, label, keypt = data_queue.dequeue()
  raw = tf.read_file(img_file)
  img = tf.image.decode_jpeg(raw)
#     label = tf.string_to_number(label, tf.int32)
  resized_img = tf.image.resize_images(img, tf.constant([227, 227]))
  fliped_img = tf.image.random_flip_left_right(resized_img)
  distorted_img = tf.image.random_brightness(fliped_img, max_delta=50)
  distorted_img = tf.image.random_contrast(distorted_img, lower=0.2, upper=1.8)
  float_img = tf.image.per_image_standardization(distorted_img)
  float_img.set_shape([227, 227, 3])
  images, labels, keypts = tf.train.shuffle_batch([float_img, label, keypt],
                                                  batch_size=batch_size,
                                                  capacity=1000 + 3 * batch_size,
                                                  min_after_dequeue=1000)
  return images, labels, keypts


def inputs(batch_size):
  img_list = []
  label_list = []
  keypts_list = []
  i = 0
  for folder in os.listdir('test'):
    if folder == '.DS_Store':
      continue
    cur_list = ['test/' + folder + '/' + img 
                for img in os.listdir('test/' + folder) if img.endswith('.jpg')]
    label_list.extend([i] * len(cur_list))
    keypts_list.extend(get_keypts(folder, cur_list, train=False))
    img_list.extend(cur_list)
    i += 1
      
  data_queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.int32, tf.float32], shapes=[[],[],[36]])
  enqueue_op = data_queue.enqueue_many([img_list, label_list, keypts_list])
  qr = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
  tf.train.add_queue_runner(qr)
  
  img_file, label, keypt = data_queue.dequeue()
  raw = tf.read_file(img_file)
  img = tf.image.decode_jpeg(raw)
  
  resized_img = tf.image.resize_images(img, tf.constant([227, 227]))
  float_img = tf.image.per_image_standardization(resized_img)
  float_img.set_shape([227, 227, 3])
  images, labels, keypts  = tf.train.batch([float_img, label, keypt],
                                            batch_size=batch_size,
                                            capacity=300 + 2 * batch_size)
  return images, labels, keypts