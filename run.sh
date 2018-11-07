#!/bin/bash

####### data information ####
if [ $(hostname) = 'dgx1' ];  then
    # running code on the dgx1
    ROOT_DIR="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images"
    IMAGE_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.001
    PYTHON=py_gxdai

elif [ $(hostname) = 'aduae266-lap' ]; then
   # running code one nyu machine
    ROOT_DIR="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images"
    IMAGE_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/home/gxdai/MMVC_LARGE2/Guoxian_Dai/data/cub_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
    PYTHON=python

elif [ $(hostname) = 'institute01' ];  then
    # running code on the dgx1
    ROOT_DIR="/raid/Guoxian_Dai/cub_2011/CUB_200_2011/images"
    IMAGE_TXT="/raid/Guoxian_Dai/cub_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/raid/Guoxian_Dai/cub_2011/CUB_200_2011/train_test_split.txt" LABEL_TXT="/raid/Guoxian_Dai/cub_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.001
    PYTHON=py_gxdai

elif [ $(hostname) = 'uranus' ];  then
    echo "running code on the uranus"
    echo "=========================="
    echo "Activate virutalenv"

    source $HOME/py2/bin/activate
    source $HOME/tf_path

    echo "START >>>>>>>>>>>>>>>>>>>"

    ROOT_DIR="/data1/Guoxian_Dai/CUB_200_2011/images"
    IMAGE_TXT="/data1/Guoxian_Dai/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/data1/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/data1/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
    TRAIN_BATCH_SIZE=64
    PYTHON=py_gxdai

elif [ $(hostname) = 'MARS' ];  then
    echo "running code on 102"
    echo "Activate anaconda"
    ANACONDA3=/home/gxdai/anaconda3

    source ${ANACONDA3}/etc/profile.d/conda.sh
    source ${ANACONDA3}/bin/activate py37

    ROOT_DIR="/data/Guoxian_Dai/CUB_200_2011/images"
    IMAGE_TXT="/data/Guoxian_Dai/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/data/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/data/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
    TRAIN_BATCH_SIZE=64
    PYTHON=py_gxdai


elif [ $(hostname) = 'MERCURY' ];  then
    echo "running code on 101"
    echo "Activate anaconda"
    ANACONDA3=/home/gxdai/anaconda3

    source ${ANACONDA3}/etc/profile.d/conda.sh
    source ${ANACONDA3}/bin/activate py37

    ROOT_DIR="/data/Guoxian_Dai/CUB_200_2011/images"
    IMAGE_TXT="/data/Guoxian_Dai/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/data/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/data/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.0001
    TRAIN_BATCH_SIZE=32
    PYTHON=py_gxdai

elif [ $(hostname) = 'gxdai-Precision-7920-Tower' ];  then
    # running code on the dgx1

    ROOT_DIR="/home/gxdai/cub_2011/CUB_200_2011/images"
    IMAGE_TXT="/home/gxdai/cub_2011/CUB_200_2011/images.txt"
    TRAIN_TEST_SPLIT_TXT="/home/gxdai/cub_2011/CUB_200_2011/train_test_split.txt"
    LABEL_TXT="/home/gxdai/cub_2011/CUB_200_2011/image_class_labels.txt"
    LEARNING_RATE=0.001
    PYTHON=python

fi

GPU_ID=$1
LOSS_TYPE="contrastive_loss"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
                        --optimizer "momentum" \
                        --train_batch_size $TRAIN_BATCH_SIZE \
                        --momentum 0.9 \
                        --learning_rate $LEARNING_RATE \
                        --learning_rate_decay_type "fixed" \
                        --loss_type $LOSS_TYPE \
                        --margin 1.0 \
                        --root_dir $ROOT_DIR \
                        --image_txt $IMAGE_TXT \
                        --train_test_split_txt $TRAIN_TEST_SPLIT_TXT \
                        --label_txt $LABEL_TXT \
                        --focal_decay_factor "1000000000.0" \
                        --display_step 20 \
                        --eval_step 30 \
                        --embedding_size 128 \
                        --num_epochs_per_decay 5 #> ${LOSS_TYPE}.txt 2>&1
                        #--with_regularizer

# Explannation for 2 and 1, file descriptor
# 2: stderr
# 1: stdout
# >file 2>&1: we are doing redirecting stdout 1 to file, meanwhile redirecting stderr 2 to the same place as stdout 1


if [ 0 -eq 1 ]; then
	--num_epochs1 200 \
	--batch_size 64 \
	--restore_ckpt 1 \
	--evaluation 1 \
	--ckpt_dir "./models/scratch/momentumOptimizer" \
	--dn_train 1 \
	--dn_test 1 \
	--weightFile "./models/scratch/momentumOptimizer/models-5" \
	--targetNum 110000
fi
# --weightFile "./models/old/my_model.ckpt-57" \
