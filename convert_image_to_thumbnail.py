import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.manifold import TSNE
import time



from dataset import generate_half_split_list, CarDataset
from utils import create_thumbnail_image # FOR CUB dataset
"""
##  CUB200-2011 

root_dir ="/data1/Guoxian_Dai/CUB_200_2011/images"
image_txt ="/data1/Guoxian_Dai/CUB_200_2011/images.txt"
train_test_split_txt ="/data1/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
label_txt ="/data1/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"
offset = 1
train_img_list, train_label_list, test_img_list, \
        test_label_list = generate_half_split_list(root_dir, image_txt,
                                                   train_test_split_txt,
                                                   label_txt, offset
                                                   )
output_dir = './cub200_thumbnail'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
size = (32, 32)
for i, img in enumerate(test_img_list):
    if i % 100 == 0:
        print("counter = {:5d}".format(i))
    split_path = img.split('/')
    image_name = split_path[-1]
    class_name = split_path[-2]
    output_path = os.path.join(output_dir, class_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    image_base_name = image_name.split('.')[0]
    new_image_name = image_base_name + '.png'
    outfile = os.path.join(output_path, new_image_name)
    create_thumbnail_image(img, outfile, size=size)
"""
## CAR196

root_dir ="/data1/Guoxian_Dai/car196"
image_txt = "/data1/Guoxian_Dai/car196/cars_annos.mat" 
train_test_split_txt ="/data1/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
label_txt ="/data1/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"
offset = 1

car_dataset = CarDataset(root_dir, image_txt)

train_img_list = car_dataset.train_img_list
train_label_list = car_dataset.train_label_list
test_img_list = car_dataset.test_img_list
test_label_list = car_dataset.test_label_list
output_dir = './car196_thumbnail'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
"""
print(len(train_img_list))
print(train_img_list[-2:])
print(len(train_label_list))
print(train_label_list[-2:])
print(len(test_img_list))
print(test_img_list[-2:])
print(len(test_label_list))
print(test_label_list[-2:])
sys.exit()
"""

size = (128, 128)
for i, img in enumerate(test_img_list):
    if i % 100 == 0:
        print("counter = {:5d}".format(i))
    split_path = img.split('/')
    image_name = split_path[-1]
    image_base_name = image_name.split('.')[0]
    new_image_name = image_base_name + '.png'
    outfile = os.path.join(output_dir, new_image_name)
    create_thumbnail_image(img, outfile, size=size)

