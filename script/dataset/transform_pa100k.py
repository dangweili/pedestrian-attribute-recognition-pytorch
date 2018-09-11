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
    dataset['description'] = 'pa100k'
    dataset['root'] = './dataset/pa100k/data/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(26)
    # load ANNOTATION.MAT
    data = loadmat(open('./dataset/pa100k/annotation.mat', 'r'))
    for idx in range(26):
        dataset['att_name'].append(data['attributes'][idx][0][0])

    for idx in range(80000):
        dataset['image'].append(data['train_images_name'][idx][0][0])
        dataset['att'].append(data['train_label'][idx, :].tolist())

    for idx in range(10000):
        dataset['image'].append(data['val_images_name'][idx][0][0])
        dataset['att'].append(data['val_label'][idx, :].tolist())
    
    for idx in range(10000):
        dataset['image'].append(data['test_images_name'][idx][0][0])
        dataset['att'].append(data['test_label'][idx, :].tolist())

    with open(os.path.join(save_dir, 'pa100k_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    partition['weight_train'] = []
    # load ANNOTATION.MAT
    data = loadmat(open('./dataset/pa100k/annotation.mat', 'r'))
    train = range(80000) 
    val = [i+80000 for i in range(10000)]
    test = [i+90000 for i in range(10000)]
    trainval = train + val
    partition['train'].append(train)
    partition['val'].append(val)
    partition['trainval'].append(trainval)
    partition['test'].append(test)
    # weight
    train_label = data['train_label'].astype('float32')
    trainval_label = np.concatenate((data['train_label'], data['val_label']), axis=0).astype('float32')
    weight_train = np.mean(train_label==1, axis=0).tolist()
    weight_trainval = np.mean(trainval_label==1, axis=0).tolist()

    partition['weight_trainval'].append(weight_trainval)
    partition['weight_train'].append(weight_train)

    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pa100k dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/pa100k/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/pa100k/pa100k_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
