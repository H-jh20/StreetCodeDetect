import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from train import trainer
from data import StreetDatasets
from test import run_test

def make_arg():
    parser = argparse.ArgumentParser("res_18 train")
    # parser.add_argument("-f","--")
    parser.add_argument("-i", "--img_dir", default="./datasets", type=str, help="datasets path")
    parser.add_argument("-s", "--save_dir", default="./outputs", type=str, help="where is outputs")
    parser.add_argument("-m","--mode",default="train",type=str,help="train/val/test")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for continue")
    parser.add_argument("-cd", "--cuda", default=True, type=bool, help="is gpu?")
    parser.add_argument("-b","--batch_size",default=8,type=int,help="")
    parser.add_argument("--epoch",default=30,type=int,help="")
    parser.add_argument("--lr",default=0.001,type=float,help="")
    parser.add_argument("--savename",default="aug",type=str,help="")

    return parser.parse_args()
def make_data(args):
    transform = transforms.Compose([transforms.Resize((250,250)),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ColorJitter(0.3, 0.3, 0.2),#亮度0.3，对比度0.3，饱和度0.2
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    dataset = StreetDatasets(args,transform,mode = "train")
    dataloder = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)

    valdataset = StreetDatasets(args,transform,mode = "val")
    val_dataloder = DataLoader(valdataset,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    return dataloder,val_dataloder


if __name__ == '__main__':
    args = make_arg()
    dataloder ,val_dataloder= make_data(args)
    trainer = trainer(args,dataloder,val_dataloder)
    trainer.train()
    run_test()
