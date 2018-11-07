# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random



def create_dict(img_list, label_list):
    """
    args:
        img_list: [img_path1, img_path2, ...]
        label_list: [label1, label2, ....]
    returns:
        imgs_dict: {1: [...], 2: [...]
    """
    assert len(img_list) == len(label_list), \
            "the number of image and label does not match"
    img_dict = {}
    for img, label in zip(img_list, label_list):
        if label not in img_dict:
            img_dict[label] = [img]
        else:
            img_dict[label].append(img)

    return img_dict, len(img_dict.keys())



def create_triplet(img_dict, batch_size, class_num):
    # image dict
    batch_label = np.random.permutate(class_num)
    batch_label = batch_label[:batch_size]









def create_dataset(train_img_list, train_label_list, \
        test_img_list, test_label_list, batch_size):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    images: a batch of images
    labels: a batch of labels
    train_init_op: initialize iterator for training data.
    test_init_op: initialize iterator for testing data.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  # create dataset for training data
  train_dataset = tf.data.Dataset.from_tensor_slices((train_img_list, train_label_list))
  # transform dataset
  train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=8)
  # shuffle the dataset
  train_dataset = train_dataset.shuffle(buffer_size=6000)
  train_dataset = train_dataset.batch(batch_size)


  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, \
          train_dataset.output_shapes)

  images, labels = iterator.get_next()




  # create dataset for test data
  test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_label_list))
  # transform dataset
  test_dataset = test_dataset.map(map_func=parse_function, num_parallel_calls=8)
  # shuffle the dataset
  test_dataset = test_dataset.batch(batch_size)

  # initialize iterator
  train_init_op = iterator.make_initializer(train_dataset)
  test_init_op = iterator.make_initializer(test_dataset)

  return images, labels, train_init_op, test_init_op

def generate_list(root_dir, image_txt, train_test_split_txt, label_txt, offset=1):
    """
    Args:
        root_dir: the root directory for image data.
        image_txt: a txt file listing all the image paths.
        train_test_split_txt: a txt file listing the image set (training, or testing),
                                denoting whether the image is for training or testing.
        label_txt:  a txt file list all the image label.
        offset:     the started value of label.

    Returns:
        train_img_list: [train_img_path1, train_img_path2, ...]
        train_label_list: [train_label1, train_label2, ....]
        test_img_list: [test_img_path1, test_img_path2, ...]
        test_label_list: [test_label1, test_label2, ...]
    """

    # check the file existence.
    # all(): logic and across all the elements in a list
    if not (os.path.isdir(root_dir) and all(map(os.path.isfile, \
            [image_txt, train_test_split_txt, label_txt]))):
        raise IOError("Please check the files or paths existence")

    train_img_list = []
    train_label_list =  []      # the label starts from 1
    test_img_list = []
    test_label_list = []        # the label starts from 1

    # read all the file
    # fl_txt is a list for file pointers
    # with map(open, [image_txt, train_test_split_txt, label_txt]) as fl_txt:
    fl_txt = map(open, [image_txt, train_test_split_txt, label_txt])
    # read each file accordingly
    for img_path, set_label, label in zip(*fl_txt):
        # check if the image id is match, if not, continue
        img_path = img_path.rstrip('\n').split(' ')
        set_label = set_label.rstrip('\n').split(' ')
        label = label.rstrip('\n').split(' ')
        if not (img_path[0] == set_label[0] == label[0]):
            print(img_path, set_label, label)
            continue

        # put all the data into list
        if set_label[-1] == '1': # train set label is 1
            train_img_list.append(os.path.join(root_dir, img_path[-1]))
            train_label_list.append(int(label[-1]) - offset)
        elif set_label[-1] == '0': # testing set label is 0
            test_img_list.append(os.path.join(root_dir, img_path[-1]))
            test_label_list.append(int(label[-1]) - offset)

        else:
            print("The set is unclear")
            print(img_path, set_label, label)
            continue


    # shuffle the training list
    merged_list = list(zip(train_img_list, train_label_list))
    random.shuffle(merged_list)
    train_img_list, train_label_list = zip(*merged_list)
    # convert tuple to list
    train_img_list, train_label_list = \
            list(train_img_list), list(train_label_list)

    return train_img_list, train_label_list, test_img_list, test_label_list

def generate_half_split_list(root_dir, image_txt, train_test_split_txt, label_txt, offset=1):
    """
    Args:
        root_dir: the root directory for image data.
        image_txt: a txt file listing all the image paths.
        train_test_split_txt: a txt file listing the image set (training, or testing),
                                denoting whether the image is for training or testing.
        label_txt:  a txt file list all the image label.
        offset:     the started value of label.

    Returns:
        train_img_list: [train_img_path1, train_img_path2, ...]
        train_label_list: [train_label1, train_label2, ....]
        test_img_list: [test_img_path1, test_img_path2, ...]
        test_label_list: [test_label1, test_label2, ...]
    """

    # check the file existence.
    # all(): logic and across all the elements in a list
    if not (os.path.isdir(root_dir) and all(map(os.path.isfile, \
            [image_txt, train_test_split_txt, label_txt]))):
        raise IOError("Please check the files or paths existence")

    train_img_list = []
    train_label_list =  []      # the label starts from 1
    test_img_list = []
    test_label_list = []        # the label starts from 1

    # read all the file
    # fl_txt is a list for file pointers
    # with map(open, [image_txt, train_test_split_txt, label_txt]) as fl_txt:
    fl_txt = map(open, [image_txt, train_test_split_txt, label_txt])
    # read each file accordingly
    for img_path, set_label, label in zip(*fl_txt):
        # check if the image id is match, if not, continue
        img_path = img_path.rstrip('\n').split(' ')
        set_label = set_label.rstrip('\n').split(' ')
        label = label.rstrip('\n').split(' ')
        if not (img_path[0] == set_label[0] == label[0]):
            print(img_path, set_label, label)
            continue

        # put all the data into list
        if int(label[-1]) <= 100: # train set with the first half classes
            train_img_list.append(os.path.join(root_dir, img_path[-1]))
            train_label_list.append(int(label[-1]) - offset)
        elif int(label[-1]) <= 200: # testing set using the second half classes
            test_img_list.append(os.path.join(root_dir, img_path[-1]))
            test_label_list.append(int(label[-1]) - offset - 100)

        else:
            print("The set is unclear")
            print(img_path, set_label, label)
            continue


    # shuffle the training list
    merged_list = list(zip(train_img_list, train_label_list))
    random.shuffle(merged_list)
    train_img_list, train_label_list = zip(*merged_list)
    # convert tuple to list
    train_img_list, train_label_list = \
            list(train_img_list), list(train_label_list)




    print("training image number is {}".format(len(train_img_list)))
    print("testing image number is {}".format(len(test_img_list)))
    return train_img_list, train_label_list, test_img_list, test_label_list





def parse_function(file_path, label):
    """
    Args:
        file_path: the full path of a single image

    Returns:
        image: image tensor
    """

    # read image into string
    image_string = tf.read_file(file_path)

    # decode the string into image, default channels = 3
    image_resized = tf.image.resize_images(tf.image.decode_jpeg(image_string, channels=3), (299, 299))
    image_resized = tf.divide(tf.cast(image_resized, tf.float32), 255.)
    image_rescaled = tf.subtract(image_resized, 0.5)
    image_rescaled = tf.multiply(image_rescaled, 2.0)


    return image_rescaled, label





if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root_dir = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images'
    image_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt'
    train_test_split_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    label_txt = '/raid/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
    batch_size = 100


    # generating list file
    train_img_list, train_label_list, test_img_list, test_label_list \
          = generate_list(root_dir, image_txt, train_test_split_txt, label_txt)

    # create iterator
    images, labels, train_init_op, test_init_op = \
            create_dataset(train_img_list, train_label_list, \
            test_img_list, test_label_list, batch_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print('init train op')
        sess.run(train_init_op)

        print("training data load")
        for i in range(20):
            print("The {}-th batch".format(i))
            batch_images, batch_labels = sess.run([images, labels])

            print('batch_images.shape = {}'.format(batch_images.shape))
            print('batch_labels.shape = {}'.format(batch_labels.shape))
        print('init test op')
        sess.run(test_init_op)
        print('testing data load')
        for i in range(20):
            print('The {}-th batch'.format(i))
            batch_images, batch_labels = sess.run([images, labels])
            print('batch_images.shape = {}'.format(batch_images.shape))
            print('batch_labels.shape = {}'.format(batch_labels.shape))

