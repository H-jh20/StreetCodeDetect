import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import os
import json
from PIL import Image
import numpy as np


class StreetDatasets(Dataset):
    def __init__(self, args,transform=None,mode = "train"):
        self.mode = mode
        self.img_dir = args.img_dir + "/{}".format(mode)
        self.img_path = os.listdir(self.img_dir+"/mchar_{}".format(mode))
        self.label_path = self.img_dir+"/label/mchar_{}.json".format(mode)


        f = open(self.label_path,"r")
        self.all_label = json.load(f)
        self.transform = transform
    def __getitem__(self, idx):
        img_name = self.img_path[idx]

        img = Image.open(self.img_dir+"/mchar_{}/".format(self.mode)+img_name)
        if self.transform is not None:
            img = self.transform(img)
        img_info =  self.all_label[img_name]
        target = img_info["label"]

        ###图像中门牌号位数最多为6，因为返回的标签数量必须一致，所以不足6为的补足6位
        target = np.array(target, dtype=np.int)
        target = list(target) + (6 - len(target)) * [10]

        target = torch.from_numpy(np.array(target[:6]))
        return img,target

    def __len__(self):
        return len(self.img_path)

class TestStreetSet(Dataset):
    def __init__(self,args,transform=None):
        super(TestStreetSet, self).__init__()
        self.img_dir = args.img_dir+"/test/mchar_test_a"
        self.img_path = os.listdir(self.img_dir)
        self.img_path = sorted(self.img_path)
        self.transform = transform
    def __getitem__(self, idx):
        img = Image.open(self.img_dir+"/"+self.img_path[idx])
        if self.transform is not None:
            img = self.transform(img)

        return img,self.img_path[idx]
    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("res_18 train")
    # parser.add_argument("-f","--")
    parser.add_argument("-i", "--img_dir", default="./datasets", type=str, help="datasets path")
    parser.add_argument("-s", "--save_dir", default="./outputs", type=str, help="where is outputs")
    parser.add_argument("-m","--mode",default="train",type=str,help="train/val/test")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for continue")
    parser.add_argument("-d", "--devices", default=True, type=bool, help="is gpu?")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    # transforms.Pad(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])

    data = StreetDatasets(args,transform)
    ddd = DataLoader(data,batch_size=4,shuffle=False,num_workers=0,drop_last=False)

    label_num = []
    for sample in ddd:
        print(sample[0])
