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
    dataset['description'] = 'peta'
    dataset['root'] = './dataset/peta/images/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(35)
    # load PETA.MAT
    data = loadmat(open('./dataset/peta/PETA.mat', 'r'))
    for idx in range(105):
        dataset['att_name'].append(data['peta'][0][0][1][idx,0][0])

    for idx in range(19000):
        dataset['image'].append('%05d.png'%(idx+1))
        dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())
    with open(os.path.join(save_dir, 'peta_dataset.pkl'), 'w+') as f:
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
    # load PETA.MAT
    data = loadmat(open('./dataset/peta/PETA.mat', 'r'))
    for idx in range(5):
        train = (data['peta'][0][0][3][idx][0][0][0][0][:,0]-1).tolist()
        val = (data['peta'][0][0][3][idx][0][0][0][1][:,0]-1).tolist()
        test = (data['peta'][0][0][3][idx][0][0][0][2][:,0]-1).tolist()
        trainval = train + val
        partition['train'].append(train)
        partition['val'].append(val)
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        # weight
        weight_trainval = np.mean(data['peta'][0][0][0][trainval, 4:].astype('float32')==1, axis=0).tolist()
        weight_train = np.mean(data['peta'][0][0][0][train, 4:].astype('float32')==1, axis=0).tolist()
        partition['weight_trainval'].append(weight_trainval)
        partition['weight_train'].append(weight_train)
    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="peta dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/peta/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/peta/peta_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
