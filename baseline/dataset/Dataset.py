import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import cPickle as pickle
import copy

class AttDataset(data.Dataset):
    """
    person attribute dataset interface
    """
    def __init__(
        self, 
        dataset,
        partition,
        split='train',
        partition_idx=0,
        transform=None,
        target_transform=None,
        **kwargs):
        if os.path.exists( dataset ):
            self.dataset = pickle.load(open(dataset))
        else:
            print dataset + ' does not exist in dataset.'
            raise ValueError
        if os.path.exists( partition ):
            self.partition = pickle.load(open(partition))
        else:
            print partition + ' does not exist in dataset.'
            raise ValueError
        if not self.partition.has_key(split):
            print split + ' does not exist in dataset.'
            raise ValueError
        
        if partition_idx > len(self.partition[split])-1:
            print 'partition_idx is out of range in partition.'
            raise ValueError

        self.transform = transform
        self.target_transform = target_transform

        # create image, label based on the selected partition and dataset split
        self.root_path = self.dataset['root']
        self.att_name = [self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]
        self.image = []
        self.label = []
        for idx in self.partition[split][partition_idx]:
            self.image.append(self.dataset['image'][idx])
            label_tmp = np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']].tolist()
            self.label.append(label_tmp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname, target = self.image[index], self.label[index]
        # load image and labels
        imgname = os.path.join(self.dataset['root'], imgname)
        img = Image.open(imgname)
        if self.transform is not None:
            img = self.transform( img )
        
        # default no transform
        target = np.array(target).astype(np.float32)
        target[target == 0] = -1
        target[target == 2] = 0
        if self.target_transform is not None:
            target = self.transform( target )

        return img, target

    # useless for personal batch sampler
    def __len__(self):
        return len(self.image)


