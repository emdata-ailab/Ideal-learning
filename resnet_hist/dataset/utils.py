from __future__ import print_function
from __future__ import division

from torchvision import transforms
import PIL.Image
from PIL import Image
import torch
import random


def std_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).mean(dim=1)


class Identity():  # used for skipping transforms
    def __call__(self, im):
        return im


class print_shape():
    def __call__(self, im):
        print(im.size)
        return im


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class pad_shorter():
    def __call__(self, im):
        h, w = im.size[-2:]
        s = max(h, w)
        new_im = PIL.Image.new("RGB", (s, s))
        new_im.paste(im, ((s - h) // 2, (s - w) // 2))
        return new_im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
                         tensor - self.in_range[0]
                 ) / (
                         self.in_range[1] - self.in_range[0]
                 ) * (
                         self.out_range[1] - self.out_range[0]
                 ) + self.out_range[0]
        return tensor

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


def default_loader(path):
    return Image.open(path).convert('RGB')

def Generate_transform_Dict():

    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]

    transform_dict = {}

    transform_dict['rand-crop'] = transforms.Compose(
        [
            transforms.Resize(resnet_sz_resize),
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['center-crop'] = transforms.Compose(
        [
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['center-crop-90'] = transforms.Compose(
        [
            RotateAnticlockwise(90),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['center-crop-180'] = transforms.Compose(
        [
            RotateAnticlockwise(180),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['center-crop-270'] = transforms.Compose(
        [
            RotateAnticlockwise(270),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )
    
    transform_dict['center-crop-rand'] = transforms.Compose(
        [
            RotateAnticlockwise(rotate=[0,90,180,270]),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )
    
    transform_dict['resize'] = transforms.Compose(
        [
            transforms.Resize(resnet_sz_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['rand-crop-90'] = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            RotateAnticlockwise(90),
            transforms.Resize(resnet_sz_resize),
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    ) 

    transform_dict['rand-crop-180'] = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            RotateAnticlockwise(180),
            transforms.Resize(resnet_sz_resize),
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['rand-crop-270'] = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            RotateAnticlockwise(270),
            transforms.Resize(resnet_sz_resize),
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['rand-crop-rand'] = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            RotateAnticlockwise(rotate=[0,90,180,270]),
            transforms.Resize(resnet_sz_resize),
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['aug_train'] = transforms.Compose(
        [
            # 随机裁剪：在随机位置裁剪图像，输出的尺寸为 input_size x input_size
            transforms.RandomResizedCrop(resnet_sz_resize, scale=(0.8, 1.0), ratio=(0.75, 1.33)),

            # 随机水平翻转：以 50% 的概率水平翻转图像
            transforms.RandomHorizontalFlip(),

            # 随机垂直翻转：以 50% 的概率垂直翻转图像
            transforms.RandomVerticalFlip(),

            # 颜色抖动：随机改变亮度、对比度、饱和度和色调
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            # 随机旋转：随机旋转图像，范围在 -30 到 30 度之间
            transforms.RandomRotation(degrees=30),

            # 随机缩放和平移：通过仿射变换随机缩放和平移图像
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )


    transform_dict['hist_train'] = transforms.Compose(
        [
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )

    transform_dict['hist_test'] = transforms.Compose(
        [
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )
    
    return transform_dict


def make_transform(is_train=True, is_inception=False):

    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std)
        ]
    )

    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
        [
            RGBToBGR(),
            transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.Resize(inception_sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            transforms.Normalize(mean=inception_mean, std=inception_std)
        ])

    return inception_transform if is_inception else resnet_transform

