import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.manifold import TSNE

# FOR CUB200-2011
print("Loading data")
cub200 = np.load('cub200_fea_label_path.npy')
cub200 = cub200.item()
feature_set = cub200.get('feature')
label_set = cub200.get('label')
path_set = cub200.get('path')
print("processing TSNE")
"""
model = TSNE()
fea_reduce = model.fit_transform(feature_set)
np.save('tsne_fea_cub.npy', fea_reduce)
"""
print("FINISH TSNE")
fea_reduce = np.load('tsne_fea_cub.npy')
x_min = np.min(fea_reduce[:,0])
x_max = np.max(fea_reduce[:,0])

y_min = np.min(fea_reduce[:,1])
y_max = np.max(fea_reduce[:,1])


print(x_min, x_max, y_min, y_max)
# (-86.09199, 59.1048, -70.07758, 74.11734)
box1 = [-87, -77, -12, -6]
box2 = [-52, -42, -51, -43]
box3 = [10, 21, 61, 72]
box4 = [44, 52, -37, -28]
box_set = [box1, box2, box3, box4]

def plot_rect(ax, x_min, x_max, y_min, y_max, step=0.02):

    x_axis = np.arange(x_min, x_max, step)
    y_axis = np.arange(y_min, y_max, step)

    ax.plot(x_axis, np.full_like(x_axis, y_min), 'r-')
    ax.plot(x_axis, np.full_like(x_axis, y_max), 'r-')
    ax.plot(np.full_like(y_axis, x_min), y_axis, 'r-')
    ax.plot(np.full_like(y_axis, x_max), y_axis, 'r-')

    # return ax

def get_box_feature(fea_reduce, x_min, x_max, y_min, y_max):
    index_x = np.where(np.logical_and(fea_reduce[:,0] < x_max, fea_reduce[:, 0] > x_min))
    index_x = index_x[0]
    # print(index_x.shape)
    fea_sub_set = fea_reduce[index_x, 1]
    index_y = np.where(np.logical_and(fea_sub_set < y_max, fea_sub_set > y_min))
    index_y = index_y[0]
    index_xy = index_x[index_y]

    return index_xy



def get_sub_set(fea_reduce, path_set, index):
    index = index.astype(int)
    sub_fea = fea_reduce[index, :]
    sub_path = [path_set[idx] for idx in index]

    return sub_fea, sub_path








print(fea_reduce.shape)
path_set = [path.replace('/data1/Guoxian_Dai/CUB_200_2011/images', './cub200_thumbnail').replace('.jpg', '.png') for path in path_set]



# This is for subfigure

box_id = 3
box_set = [box_set[box_id]]

index_xy = get_box_feature(fea_reduce, box_set[0][0], box_set[0][1], box_set[0][2], box_set[0][3])

fea_reduce, path_set = get_sub_set(fea_reduce, path_set, index_xy)
print(fea_reduce.shape)
print(len(path_set))







def main():
    x = fea_reduce[:,0]
    y = fea_reduce[:,1]
    # image_path = get_sample_data('ada.png')
    image_path = path_set
    fig, ax = plt.subplots()
    # plot_rect(ax, -50, 50, -50, 50)
    # plot_rect(ax, -87, -77, -12, -6)
    for box in box_set:
        plot_rect(ax, box[0], box[1], box[2], box[3])

    # zoom = 0.06 For the whole image
    # imscatter(x, y, image_path, zoom=0.06, ax=ax)

    # zoom = 0.1 For the first boxes
    imscatter(x, y, image_path, zoom=0.2, ax=ax)
    ax.plot(x, y, 'o')
    ax.axis('off')
    plt.show()

def imscatter(x, y, image_set, ax=None, zoom=1):
    def read_img(ax, image):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            # Likely already an array...
            pass

        return image

    # im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    counter = 0
    for x0, y0, image in zip(x, y, image_set):
        if counter % 100 == 0:
            print(counter, image)

        image = read_img(ax, image)
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        counter += 1

    # plot_rect(ax, -87, -77, -12, -6)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

main()
