import torch.nn
from torchvision import models
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from model import SVHDmodule
import numpy as np


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


class trainer(object):
    def __init__(self, args, dataloder, val_dataloder):
        self.args = args
        self.cuda = args.cuda
        self.lr = args.lr
        self.epoches = args.epoch
        self.ckpt = args.ckpt
        self.dataloder = dataloder
        self.val_dataloder = val_dataloder
        self.build_model()
        save_dir = "exp/{}".format(self.args.savename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.summary = SummaryWriter(save_dir+ str(self.lr))

    def build_model(self):
        model = SVHDmodule()
        if self.ckpt:
            self.model = torch.load(self.ckpt)
        else:
            self.model = model
        if self.cuda:
            self.model.cuda()
        self.model.eval()

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda epoch:0.1*epoch)
        self.loss_func = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(1)

    def train(self):
        self.before_train()
        print("start train")
        mix_train_loss = 10
        max_train_accuracy = 0
        mix_val_loss = 10
        max_val_accuracy = 0
        for epoch in range(self.epoches):
            self.model.train()
            loop = tqdm(enumerate(self.dataloder), total=len(self.dataloder))
            for idx, (img, target) in loop:
                img = img.cuda()
                target = target.cuda()

                out_put1, out_put2, out_put3, out_put4, out_put5, out_put6 = self.model(img)
                loss = self.loss_func(out_put1, target[:, 0].long()) + self.loss_func(out_put2, target[:, 1].long()) + \
                       self.loss_func(out_put3, target[:, 2].long()) + self.loss_func(out_put4, target[:, 3].long()) + \
                       self.loss_func(out_put5, target[:, 4].long()) + self.loss_func(out_put6, target[:, 5].long())

                if (idx % 1000 == 0):
                    self.summary.add_scalar("loss", loss, epoch * self.epoches + idx / 1000)

                if loss < mix_train_loss:
                    mix_train_loss = loss.item()
                    self.save_model("minTL", epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 计算准确率
                train_predict = torch.stack((out_put1.max(1)[1], out_put2.max(1)[1],
                                           out_put3.max(1)[1], out_put4.max(1)[1],
                                           out_put5.max(1)[1],out_put6.max(1)[1]),1)
                predict_number = torch.ge(torch.sum(train_predict.eq(target),dim=1).squeeze(),6).data.cpu().numpy()
                accuracy = np.mean(predict_number)
                # 更新tqdm
                loop.set_description(f'Epoch [{epoch}/{self.epoches}]')
                loop.set_postfix(loss=loss.item(), acc=accuracy)
            # 运行验证集
            self.lr_scheduler.step()
            val_loss, val_acc = self.eval()
            self.summary.add_scalar("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            self.summary.add_scalar("val_loss", val_loss, epoch)
            self.summary.add_scalar("val_accuracy", val_acc, epoch)
            if val_loss < mix_val_loss:
                mix_val_loss = val_loss.item()
                self.save_model("minVL", epoch)
            if val_acc > max_val_accuracy:
                max_val_accuracy = val_acc.item()
                self.save_model("maxVA", epoch)
        self.summary.close()

    def save_model(self, mode, number):
        save_dir = "./model/res18_adam_{}_{}".format(self.lr,self.args.savename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.model, "{}/res18_{}_{}.pth".format(save_dir, self.epoches, mode))

    def eval(self):
        self.model.eval()
        val_loss = []
        val_accuracy = []
        for idx, (img, target) in enumerate(self.val_dataloder):
            img = img.cuda()
            target = target.cuda()
            with torch.no_grad():
                out_put1, out_put2, out_put3, out_put4, out_put5, out_put6 = self.model(img)
                loss = self.loss_func(out_put1, target[:, 0].long()) + self.loss_func(out_put2, target[:, 1].long()) + \
                       self.loss_func(out_put3, target[:, 2].long()) + self.loss_func(out_put4, target[:, 3].long()) + \
                       self.loss_func(out_put5, target[:, 4].long()) + self.loss_func(out_put6, target[:, 5].long())
                val_loss.append(loss.item())

                val_predict = torch.stack((out_put1.max(1)[1], out_put2.max(1)[1], out_put3.max(1)[1],
                                             out_put4.max(1)[1], out_put5.max(1)[1], out_put6.max(1)[1]), 1)
                predict_number = torch.ge(torch.sum(val_predict.eq(target),dim=1).squeeze(),6).data.cpu().numpy()
                val_accuracy.append(predict_number)
        return np.mean(val_loss), np.mean(val_accuracy)
