import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data import TestStreetSet
from tqdm import tqdm
import csv
from model import SVHDmodule


class Tester(object):
    def __init__(self, args, dataloder):
        self.args = args
        self.dataloder = dataloder

    def tester(self):
        self.module = torch.load(self.args.ckpt)
        self.module.eval()
        test_predict = []
        test_name = []
        with torch.no_grad():
            loop = tqdm(enumerate(self.dataloder), total=len(self.dataloder))
            for i, (img, name) in loop:
                img = img.cuda()
                out1, out2, out3, out4, out5, out6, = self.module(img)
                stack = np.concatenate([out1.data.cpu().numpy(), out2.data.cpu().numpy(), out3.data.cpu().numpy(),
                                        out4.data.cpu().numpy(), out5.data.cpu().numpy(), out6.data.cpu().numpy()],
                                       axis=1)
                test_predict.append(stack)

        test_predict = np.vstack(test_predict)
        test_predict = np.vstack([test_predict[:, :11].argmax(1), test_predict[:, 11:22].argmax(1),
                                  test_predict[:, 22:33].argmax(1), test_predict[:, 33:44].argmax(1),
                                  test_predict[:, 44:55].argmax(1), test_predict[:, 55:].argmax(1)]).T
        self.result = []
        for x in test_predict:
            self.result.append(''.join(map(str, x[x != 10])))
        self.test_name = self.dataloder.dataset.img_path
        self.submit()

    def submit(self):
        data = {"file_name": self.test_name, "file_code": self.result}
        data_df = pd.DataFrame(data)
        data_df.to_csv("res18_50_{}.csv".format(self.args.savename), index=None)

        # df_submit = pd.read_csv('./submit/test_A_sample_submit.csv')
        # df_submit['file_code'] = self.result
        # df_submit.to_csv('submit.csv', index=None)


def make_arg():
    parser = argparse.ArgumentParser("res_18 train")
    # parser.add_argument("-f","--")
    parser.add_argument("-i", "--img_dir", default="./datasets", type=str, help="datasets path")
    parser.add_argument("-s", "--save_dir", default="./outputs", type=str, help="where is outputs")
    parser.add_argument("-m", "--mode", default="test", type=str, help="train/val/test")
    parser.add_argument("-c", "--ckpt", default="./model/res18_adam_0.001_aug/res18_30_maxVA.pth", type=str,
                        help="ckpt for continue")
    parser.add_argument("-cd", "--cuda", default=True, type=bool, help="is gpu?")
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="")
    parser.add_argument("--epoch", default=10, type=int, help="")
    parser.add_argument("--savename", default="aug", type=str, help="")

    return parser.parse_args()


def make_data(args):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = TestStreetSet(args, transform)
    dataloder = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return dataloder


def run_test():
    args = make_arg()
    dataloder = make_data(args)
    tester = Tester(args, dataloder)
    tester.tester()


if __name__ == '__main__':
    run_test()
