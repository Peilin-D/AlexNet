import os
import json
import argparse
import scipy.misc as misc
import numpy as np

def scale_keypts(train):
    if train:
        path = 'train/'
        keypts_path = 'keypts_train/'
    else:
        path = 'test/'
        keypts_path = 'keypts_test/'
    for folder in os.listdir(keypts_path):
        if folder == '.DS_Store':
            continue
        for keypts_file in os.listdir(keypts_path + folder):
            if not keypts_file.endswith('.json'):
                continue
            with open(keypts_path + folder + '/' + keypts_file) as f:
                keypts_js = json.load(f)
            if len(keypts_js['people']) != 1:
                continue
            img_file = path + folder + '/' + '_'.join(keypts_file.split('_')[:-1]) + '.jpg'
            base = keypts_file.rstrip('.json')
            orig_shape = misc.imread(img_file).shape
            ratio = (227.0 / orig_shape[1], 227.0 / orig_shape[0])
            if not os.path.exists(path + folder + '/scaled_keypts'):
                os.mkdir(path + folder + '/scaled_keypts')
            with open(path + folder + '/scaled_keypts/' + base + '_scaled.json', 'wb+') as f:
                people = keypts_js['people'][0]
                keypts = people['pose_keypoints']
                x = keypts[::3]
                y = keypts[1::3]
                c = keypts[2::3]
                
                xs = list(np.array(x) * ratio[0])
                ys = list(np.array(y) * ratio[1])
                
                json.dump(zip(xs, ys), f)
        print '%s done'%folder

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action='store_true', default=False)
  args = parser.parse_args()
  scale_keypts(args.train)