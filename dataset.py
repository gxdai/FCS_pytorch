from __future__ import print_function
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image



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
        # create a dummy training list
        merge_list = list(zip(self.train_img_list, self.train_label_list))
        random.shuffle(merge_list)
        self.train_img_list_dummy, self.train_label_list_dummy = tuple(zip(*merge_list))
        

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

            sample = {'img_1': img_1, 'img_2': img_2, 'sim_label': float(label_1 == label_2)}
        else:
            img = Image.open(self.test_img_list[idx]).convert('RGB')
            label = self.test_label_list[idx]
            if self.transform:
                img = self.transform(img)

            sample = {'img': img, 'label': label}

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


 

if __name__ == '__main__':

    import os
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
    """
    print(sample['img_1'].size())
    print(sample['img_2'].numpy().shape)
    print(sample['sim_label'].size())
    """
