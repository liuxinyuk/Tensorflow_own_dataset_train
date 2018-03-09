# -*- coding: utf-8 -*-
"""
Usage:
  # From tensorflow/models/research/object_detection/tools_for_dataset/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
  ------------------------
  or change the path in this .py
  note that the datapath should include:
    -Annotations
    -JPEGImages
    -CSV_tfrecord
      -train_labels.csv
      -test_labels.csv
  restart the python kernel each time run the .py , one time for train.record ,another time for test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

'''change here'''
path='Path to data'

flags = tf.app.flags
#获取train和test的input路径（包含train_labels.csv与test_labels.csv文件）和output路径
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

#将class转换成数字
'''change here'''
def class_text_to_int(row_label):
    if row_label == 'class1':
        return 1
#    if row_label == 'class2':
#        return 2
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    #读取图片j及其信息
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    '''change here'''
    output_path = path+'\\CSV_tfrecord\\train.record'
    #output_path = path+'\\CSV_tfrecord\\test.record'
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    fullpath = os.path.join(path, 'JPEGImages')
    
    '''change here'''
    examples = pd.read_csv(path+'\\CSV_tfrecord\\train_labels.csv')
    #examples = pd.read_csv(path+'\\CSV_tfrecord\\test_labels.csv')
    
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, fullpath)
        writer.write(tf_example.SerializeToString())

    writer.close()
    #output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    
    print('Successfully created the TFRecords: {}'.format(output_path))
    


if __name__ == '__main__':
    tf.app.run()   
