from __future__ import print_function
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import scipy.io as io

class CarDataset(Dataset):
    """
    This is a customized dataset for CUB 200.
    """
    def __init__(self, root_dir, image_info_mat, transform=None, is_train=True, offset=1):
        self.root_dir = root_dir
        self.image_info_mat = image_info_mat

        self.transform = transform
        self.is_train = is_train
        self.offset = offset

        self.train_img_list, self.train_label_list, self.test_img_list, \
            self.test_label_list = self.generate_split_list()
        # shuffle training list
        self.shuffle_list()
        print("len(self.train_img_list) = {:5d}".format(len(self.train_img_list)))
        print("len(self.train_label_list) = {:5d}".format(len(self.train_label_list)))

        print("len(self.test_img_list) = {:5d}".format(len(self.test_img_list)))
        print("len(self.test_label_list) = {:5d}".format(len(self.test_label_list)))
        """
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        """

    def generate_split_list(self):
        # load mat data
        label_set = io.loadmat(self.image_info_mat)
        label_set = label_set['annotations'][0]
        # get total number of data
        total_num = label_set.size
        train_img_list = []
        train_label_list = []
        test_img_list = []
        test_label_list = []
        for i in range(total_num):
            path = os.path.join(self.root_dir, label_set[i][0][0])
            label = label_set[i][5][0][0]
            if i < 8054:
                train_img_list.append(path)
                train_label_list.append(label-1)
            else:
                test_img_list.append(path)
                test_label_list.append(label-99)

        return train_img_list, train_label_list, test_img_list, test_label_list


    def shuffle_list(self):
        """Shuflle the list"""
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        random.shuffle(merge_list)
        self.train_img_list, self.train_label_list = tuple(zip(*merge_list))

    def __len__(self):
        if self.is_train:
            return len(self.train_img_list)
        else:
            return len(self.test_img_list)

    def __getitem__(self, idx):
        if self.is_train:
            # Use PIL.Image to read image, and convert it to RGB
            img_1 = Image.open(self.train_img_list[idx]).convert('RGB')
            label_1 = self.train_label_list[idx]

            img_2 = Image.open(self.train_img_list_dummy[idx]).convert('RGB')
            label_2 = self.train_label_list_dummy[idx]

            if self.transform:
                img_1, img_2 = self.transform(img_1), self.transform(img_2)

            sample = {'img_1': img_1, 'img_2': img_2, 'label_1': label_1, 'label_2': label_2}

            # shulffe list for next round
            if idx == self.__len__()-1:
                self.shuffle_list()


        else:
            img = Image.open(self.test_img_list[idx]).convert('RGB')
            label = self.test_label_list[idx]
            if self.transform:
                img = self.transform(img)

            # sample = {'img': img, 'label': label}
            sample = {'img': img, 'label': label, 'path': self.test_img_list[idx]}

        return sample



class CubDataset(Dataset):
    """
    This is a customized dataset for CUB 200.
    """
    def __init__(self, root_dir, image_txt, train_test_split_txt, label_txt, transform=None, is_train=True, offset=1):
        self.root_dir = root_dir
        self.image_txt = image_txt
        self.train_test_split_txt = train_test_split_txt
        self.label_txt = label_txt
        self.transform = transform
        self.is_train = is_train
        self.offset = offset

        self.train_img_list, self.train_label_list, self.test_img_list, \
            self.test_label_list = generate_half_split_list(self.root_dir, self.image_txt, \
                                                            self.train_test_split_txt, \
                                                            self.label_txt, self.offset
                                                            )
        # shuffle training list
        self.shuffle_list()
        print("len(self.train_img_list) = {:5d}".format(len(self.train_img_list)))
        print("len(self.train_label_list) = {:5d}".format(len(self.train_label_list)))

        print("len(self.test_img_list) = {:5d}".format(len(self.test_img_list)))
        print("len(self.test_label_list) = {:5d}".format(len(self.test_label_list)))
        """
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        """

    def shuffle_list(self):
        """Shuflle the list"""
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        random.shuffle(merge_list)
        self.train_img_list, self.train_label_list = tuple(zip(*merge_list))

    def __len__(self):
        if self.is_train:
            return len(self.train_img_list)
        else:
            return len(self.test_img_list)

    def __getitem__(self, idx):
        if self.is_train:
            # Use PIL.Image to read image, and convert it to RGB
            img_1 = Image.open(self.train_img_list[idx]).convert('RGB')
            label_1 = self.train_label_list[idx]

            img_2 = Image.open(self.train_img_list_dummy[idx]).convert('RGB')
            label_2 = self.train_label_list_dummy[idx]

            if self.transform:
                img_1, img_2 = self.transform(img_1), self.transform(img_2)

            sample = {'img_1': img_1, 'img_2': img_2, 'label_1': label_1, 'label_2': label_2}

            # shulffe list for next round
            if idx == self.__len__()-1:
                self.shuffle_list()


        else:
            img = Image.open(self.test_img_list[idx]).convert('RGB')
            label = self.test_label_list[idx]
            if self.transform:
                img = self.transform(img)

            sample = {'img': img, 'label': label, 'path': self.test_img_list[idx]}

        return sample

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


class OnlineProductDataset(Dataset):
    """
    This is a customized dataset for CUB 200.
    """
    def __init__(self, root_dir, train_txt, test_txt, transform=None, is_train=True, offset=1):
        self.root_dir = root_dir # root directory of data
        self.train_txt = train_txt # dataset info txt
        self.test_txt = test_txt
        self.transform = transform
        self.is_train = is_train
        self.offset = offset

        self.get_train_test_split()

        # shuffle training list
        self.shuffle_list()

        """
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        """
    def get_train_test_split(self):
        self.train_img_list = []
        self.train_label_list = []
        self.test_img_list = []
        self.test_label_list = []
        # process training dataset.
        with open(self.train_txt, 'r') as fid:
            line = fid.readline()
            for line in fid:
                line = line.split(' ')
                img_path = os.path.join(self.root_dir, line[3].rstrip('\n'))
                # make sure label start from 0
                img_label = int(line[1]) - 1
                self.train_img_list.append(img_path)
                self.train_label_list.append(img_label)
        # process testing dataset.
        with open(self.test_txt, 'r') as fid:
            line = fid.readline()
            for line in fid:
                line = line.split(' ')
                img_path = os.path.join(self.root_dir, line[3].rstrip('\n'))
                # make sure label start from 0
                img_label = int(line[1]) - 11319
                self.test_img_list.append(img_path)
                self.test_label_list.append(img_label)



    def shuffle_list(self):
        """Shuflle the list"""
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        random.shuffle(merge_list)
        self.train_img_list, self.train_label_list = tuple(zip(*merge_list))

    def __len__(self):
        if self.is_train:
            return len(self.train_img_list)
        else:
            return len(self.test_img_list)

    def __getitem__(self, idx):
        if self.is_train:
            # Use PIL.Image to read image, and convert it to RGB
            img_1 = Image.open(self.train_img_list[idx]).convert('RGB')
            label_1 = self.train_label_list[idx]

            img_2 = Image.open(self.train_img_list_dummy[idx]).convert('RGB')
            label_2 = self.train_label_list_dummy[idx]

            if self.transform:
                img_1, img_2 = self.transform(img_1), self.transform(img_2)

            # sample = {'img_1': img_1, 'img_2': img_2, 'sim_label': float(label_1 == label_2)}
            sample = {'img_1': img_1, 'img_2': img_2, 'label_1': label_1, 'label_2': label_2}

            # shulffe list for next round
            if idx == self.__len__()-1:
                self.shuffle_list()


        else:
            img = Image.open(self.test_img_list[idx]).convert('RGB')
            label = self.test_label_list[idx]
            if self.transform:
                img = self.transform(img)

            sample = {'img': img, 'label': label}

        return sample



if __name__ == '__main__':

    import os
    """
    # test cub dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 100
    root_dir ="/data1/Guoxian_Dai/CUB_200_2011/images"
    image_txt ="/data1/Guoxian_Dai/CUB_200_2011/images.txt"
    train_test_split_txt ="/data1/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
    label_txt ="/data1/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"

    transform = transforms.Compose([transforms.Resize(299),
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),        # convert PIL Image (HWC, 0-255) to (CHW, 0-1)
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                  )
    cub_dataset = CubDataset(root_dir, image_txt, train_test_split_txt, label_txt, transform=transform, is_train=False, offset=1)
    dataloader = DataLoader(dataset=cub_dataset, batch_size=64, shuffle=True, num_workers=4)
    iters = iter(dataloader)
    for _ in range(10):
        sample = next(iters)
        # sample = next(iter(cub_dataset))
        print(type(sample))
        print(sample['img'].size())
        print(sample['label'].size())
    # test online product dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 100
    root_dir ="/data/Guoxian_Dai/Stanford_Online_Products"
    train_txt ="/data/Guoxian_Dai/Stanford_Online_Products/Ebay_train.txt"
    test_txt ="/data/Guoxian_Dai/Stanford_Online_Products/Ebay_test.txt"

    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),        # convert PIL Image (HWC, 0-255) to (CHW, 0-1)
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                  )
    online_product = OnlineProductDataset(root_dir, train_txt, test_txt, transform=transform, is_train=False, offset=1)
    print("training image, label length = {:6d}, {:6d}".format(len(online_product.train_img_list), len(online_product.train_label_list)))
    print("Sample data:\n")
    # zip() return an iterator (it is not subscrible)
    print(list(zip(online_product.train_img_list, online_product.train_label_list))[:5])
    print('\n\n')
    print("testing image, label length = {:6d}, {:6d}".format(len(online_product.test_img_list), len(online_product.test_label_list)))
    print("Sample data:\n")
    print(list(zip(online_product.test_img_list, online_product.test_label_list))[:5])
    print('\n\n')
    dataloader = DataLoader(dataset=online_product, batch_size=64, shuffle=True, num_workers=4)
    iters = iter(dataloader)
    for _ in range(10):
        sample = next(iters)
        # sample = next(iter(cub_dataset))
        print(type(sample))
        print(sample['img'].size())
        print(sample['label'].size())
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 100
    root_dir ="/data1/Guoxian_Dai/car196"
    image_info_mat ="/data1/Guoxian_Dai/car196/cars_annos.mat"

    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),        # convert PIL Image (HWC, 0-255) to (CHW, 0-1)
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                  )
    car_dataset = CarDataset(root_dir, image_info_mat, transform=transform, is_train=False, offset=1)
    print("training image, label length = {:6d}, {:6d}".format(len(car_dataset.train_img_list), len(car_dataset.train_label_list)))
    print("Sample data:\n")
    # zip() return an iterator (it is not subscrible)
    print(list(zip(car_dataset.train_img_list, car_dataset.train_label_list))[:5])
    print('\n\n')
    print("testing image, label length = {:6d}, {:6d}".format(len(car_dataset.test_img_list), len(car_dataset.test_label_list)))
    print("Sample data:\n")
    print(list(zip(car_dataset.test_img_list, car_dataset.test_label_list))[:5])
    print('\n\n')
    dataloader = DataLoader(dataset=car_dataset, batch_size=64, shuffle=True, num_workers=4)
    iters = iter(dataloader)
    for _ in range(10):
        sample = next(iters)
        # sample = next(iter(cub_dataset))
        print(type(sample))
        print(sample['img'].size())
        print(sample['label'].size())
