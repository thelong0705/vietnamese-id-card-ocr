import shutil
import os
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required=True,
                help='Number of annotations')
ap.add_argument('-i', '--input', required=True,
                help='Path to the annotation folder')
ap.add_argument('-p', '--percent', required=True,
                help='percent to split annotation into train folder and val folder')

args = vars(ap.parse_args())
no_of_anno = int(args['number'])
annotations = list(range(0, no_of_anno))
random.seed(42)
random.shuffle(annotations)
num_train = int(float(args['percent'])*no_of_anno)
train_example = annotations[:num_train]
val_example = annotations[num_train:]


if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('val'):
    os.makedirs('val')
    
for i in train_example:
    shutil.copy('{}/img_{}.xml'.format(args['input'], i), 'train')
for i in val_example:
    shutil.copy('{}/img_{}.xml'.format(args['input'], i), 'val')
