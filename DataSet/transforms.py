from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random


class CovertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

class RotateAnticlockwise(object):
    def __init__(self, rotate, resample=Image.NEAREST, expand=True):
        if not isinstance(rotate, int):
            assert isinstance(rotate[0],int)
            len_rotate = len(rotate)
            self.sample_list = [random.randrange(len_rotate) for _ in range(1024)]
            self.count = 0
            # random.seed(10)
        self.rotate = rotate
        self.resample = resample
        self.expand = expand
 
    def __call__(self, img):
        if isinstance(self.rotate, int):
            rotate = self.rotate
        else:
            indx = self.sample_list[self.count % 1024]
            rotate = self.rotate[indx]
            self.count += 1
        img = img.rotate(rotate, resample=self.resample, expand=self.expand)
        return img
