import os
import numpy as np
import random
import cPickle as pickle
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'rap2'
    dataset['root'] = './dataset/rap2/RAP_dataset/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    # load RAP_annotation.mat
    data = loadmat(open('./dataset/rap2/RAP_annotation/RAP_annotation.mat', 'r'))
    dataset['selected_attribute'] = (data['RAP_annotation'][0][0][3][0,:]-1).tolist()
    for idx in range(152):
        dataset['att_name'].append(data['RAP_annotation'][0][0][2][idx][0][0])
    
    for idx in range(84928):
        dataset['image'].append(data['RAP_annotation'][0][0][0][idx][0][0])
        dataset['att'].append(data['RAP_annotation'][0][0][1][idx, :].tolist())

    with open(os.path.join(save_dir, 'rap2_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['train'] = []
    partition['val'] = []
    partition['trainval'] = []
    partition['test'] = []
    partition['weight_train'] = []
    partition['weight_trainval'] = []
    # load RAP_annotation.mat
    data = loadmat(open('./dataset/rap2/RAP_annotation/RAP_annotation.mat', 'r'))
    for idx in range(5):
        train = (data['RAP_annotation'][0][0][4][0, idx][0][0][0][0,:]-1).tolist()
        val = (data['RAP_annotation'][0][0][4][0, idx][0][0][1][0,:]-1).tolist()
        test = (data['RAP_annotation'][0][0][4][0, idx][0][0][2][0,:]-1).tolist()
        trainval = train + val
        partition['trainval'].append(trainval)
        partition['train'].append(train)
        partition['val'].append(val)
        partition['test'].append(test)
        # weight
        weight_train = np.mean(data['RAP_annotation'][0][0][1][train, :].astype('float32')==1, axis=0).tolist()
        weight_trainval = np.mean(data['RAP_annotation'][0][0][1][trainval, :].astype('float32')==1, axis=0).tolist()
        partition['weight_train'].append(weight_train)
        partition['weight_trainval'].append(weight_trainval)

    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="rap2 dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/rap2/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/rap2/rap2_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
