"""
main function.
"""
import argparse

import tensorflow as tf
import model

parser = argparse.ArgumentParser(
        description="parse arguments for focal contrastive loss module")
parser.add_argument('--root_dir', type=str,
        default='/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images')
parser.add_argument('--image_txt', type=str,
        default= '/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images.txt')
parser.add_argument('--train_test_split_txt', type=str,
        default='/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/train_test_split.txt')
parser.add_argument('--label_txt', type=str,
        default='/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/image_class_labels.txt')
parser.add_argument('--mode', type=str, default='train')


args = parser.parse_args()
def main(args):
    print(tf.__version__)
    FCS = model.FocalLoss(
            root_dir=args.root_dir, image_txt=args.image_txt,
            train_test_split_txt=args.train_test_split_txt,
            label_txt=args.label_txt)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        FCS.train(sess)


if __name__ == '__main__':
    main(args)

