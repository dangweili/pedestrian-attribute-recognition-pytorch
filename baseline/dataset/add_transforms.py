import torch
import numpy as np
import numbers
__all__ = ["AddPad", "AddCrop"]

class AddCrop(object):
    def __init__(self, size):
        self.size = size # two
        assert len(self.size) == 2
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    def __call__(self, img):
        shape = img.shape # 3*H*W
        h_high = shape[1] - self.size[0]
        w_high = shape[2] - self.size[1]
        h_start = np.random.randint(low=0, high=h_high)
        w_start = np.random.randint(low=0, high=w_high)
        return img[:, h_start: h_start+self.size[0], w_start: w_start+self.size[1]]

class AddPad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill
        if isinstance(self.padding, numbers.Number):
            self.pad_l = int(self.padding)
            self.pad_r = int(self.padding)
            self.pad_u = int(self.padding)
            self.pad_d = int(self.padding)
        elif isinstance(self.padding, (list, tuple)) and len(self.padding) == 4:
            self.pad_l = int(self.padding[0])
            self.pad_r = int(self.padding[1])
            self.pad_u = int(self.padding[2])
            self.pad_d = int(self.padding[3])
        else:
            print "The type of padding is not right."
            raise ValueError
        if self.pad_l <0 or self.pad_r < 0 or self.pad_u < 0 or self.pad_d < 0:
            raise ValueError
        if isinstance(self.fill, numbers.Number):
            self.fill_value = [self.fill]
        elif isinstance(self.fill, list):
            self.fill_value = self.fill 

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0})'.format(self.padding)

    def __call__(self, img):
        """
        Args:
            img: a 3-dimensional torch tensor with shape [R,G,B]*H*W
        Returns:
            img: a 3-dimensional padded tensor with shape [R,G,B]*H'*W'
        """
        if not (self.pad_l or self.pad_r or self.pad_u or self.pad_d):
            return img
        shape = img.shape
        img_ = torch.rand(shape[0], shape[1]+self.pad_u+self.pad_d, \
                shape[2]+self.pad_l+self.pad_r) 
        for i in range(shape[0]):
            img_[i, 0:self.pad_u, :] = self.fill_value[i%len(self.fill_value)]
            img_[i, -(self.pad_d+1):-1, :] = self.fill_value[i%len(self.fill_value)]
            img_[i, :, 0:self.pad_l] = self.fill_value[i%len(self.fill_value)]
            img_[i, :, -(self.pad_r+1):-1] = self.fill_value[i%len(self.fill_value)]
            img_[i, self.pad_u:self.pad_u+shape[1], self.pad_l:self.pad_l+shape[2]] = img[i, :, :]
        return img_
