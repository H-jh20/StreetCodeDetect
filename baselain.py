import os, sys, glob, shutil, json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

use_cuda = True


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

            img.show()
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5

def train(train_loader, model, criterion, optimizer, epoch,writer):
    # ???????????????????????????
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0].long()) + \
               criterion(c1, target[:, 1].long()) + \
               criterion(c2, target[:, 2].long()) + \
               criterion(c3, target[:, 3].long()) + \
               criterion(c4, target[:, 4].long())

        if (i % 1000 == 0):
            writer.add_scalar("loss", loss, epoch * 20 + i / 1000)

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # ???????????????????????????
    model.eval()
    val_loss = []

    # ???????????????????????????
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0].long()) + \
                   criterion(c1, target[:, 1].long()) + \
                   criterion(c2, target[:, 2].long()) + \
                   criterion(c3, target[:, 3].long()) + \
                   criterion(c4, target[:, 4].long())
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(test_loader, model, tta=10):
    use_cuda = False
    model.eval()
    test_pred_tta = None

    # TTA ??????
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

def main_train():
    train_path = glob.glob('./datasets/train/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('./datasets/train/label/mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    print(len(train_path), len(train_label))

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.RandomCrop((60, 120)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_path = glob.glob('./datasets/val/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('./datasets/val/label/mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    print(len(val_path), len(val_label))

    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    best_loss = 1000.0

    # ????????????GPU
    # use_cuda = True
    if use_cuda:
        model = model.cuda()
    writer = SummaryWriter("exp/base")
    for epoch in range(20):
        train_loss = train(train_loader, model, criterion, optimizer, epoch,writer)
        val_loss = validate(val_loader, model, criterion)

        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_accuracy", val_char_acc, epoch)

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # ????????????????????????
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './model.pt')

def main_test():
    test_path = glob.glob('./datasets/test/mchar_test_a/*.png')
    test_path.sort()
    # test_json = json.load(open('../input/test_a.json'))
    test_label = [[1]] * len(test_path)
    print(len(test_path), len(test_label))

    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                    transforms.Compose([
                        transforms.Resize((70, 140)),
                        # transforms.RandomCrop((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    # ???????????????????????????
    model = SVHN_Model1()
    model.load_state_dict(torch.load('model.pt'))
    test_predict_label = predict(test_loader, model, 1)
    print(test_predict_label.shape)

    test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    test_predict_label = np.vstack([
        test_predict_label[:, :11].argmax(1),
        test_predict_label[:, 11:22].argmax(1),
        test_predict_label[:, 22:33].argmax(1),
        test_predict_label[:, 33:44].argmax(1),
        test_predict_label[:, 44:55].argmax(1),
    ]).T

    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))

    import pandas as pd

    df_submit = pd.read_csv('./submit/mchar_sample_submit_A.csv')
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv('submit.csv', index=None)




if __name__ == '__main__':
    main_train()
    main_test()