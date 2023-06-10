import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
import numpy as np
from torchvision import transforms

# from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=transforms.Resize([128,128]), mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            # self.files = sorted([i.split('/')[-1] for i in glob.glob('/home/zhangjiansong/dataset/face/128/*')])[:8000]
            self.files = sorted([i.split('/')[-1] for i in glob.glob('/public/yuxiao/dataset/celeba/*')])[0:18000]
            print("train数据集长度：",len(self.files))
        else:
            # test
            # self.files = sorted([i.split('/')[-1] for i in glob.glob('/home/zhangjiansong/dataset/face/128/*')])[8000:]
            # self.files = sorted([i.split('/')[-1] for i in glob.glob('/public/yuxiao/dataset/celeba/*')])[18000:24000]
            self.files = sorted([i.split('/')[-1] for i in glob.glob('/public/yuxiao/dataset/celeba/*')])[28000:28002]
            print("val数据集长度：",len(self.files))
    def __getitem__(self, index):
        try:
            # image = Image.open('/home/zhangjiansong/dataset/face/128/'+self.files[index])
            image = Image.open('/public/yuxiao/dataset/celeba/' + self.files[index])
            image = to_rgb(image)
            item1 = self.transform(image)
            # image = Image.open('/home/zhangjiansong/dataset/face/mosic_128/'+self.files[index])
            image = Image.open('/public/yuxiao/dataset/mosic_celeba/' + self.files[index])
            image = to_rgb(image)
            item2 = self.transform(image)
            item = np.concatenate((item1,item2),axis=0)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


transform = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    # T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)
