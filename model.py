import torch
import torch.nn as nn
import torch.nn.functional as F

from randwire import RandWire


class Model(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, model_mode, dataset_mode, is_train, batch_size):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.model_mode = model_mode
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size

        self.num_classes = 1000
        self.dropout_rate = 0.2

        if self.dataset_mode is "CIFAR10":
            self.num_classes = 10
        elif self.dataset_mode is "CIFAR100":
            self.num_classes = 100
        elif self.dataset_mode is "IMAGENET":
            self.num_classes = 1000
        elif self.dataset_mode is "MNIST":
            self.num_classes = 10

        if self.model_mode is "CIFAR10":
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.PReLU()
            )
            #self.Max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.SW = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode, self.is_train, self.batch_size, name="1st")
            )
            #self.Up_1 = nn.Upsample(scale_factor=2)
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=3, padding=1),
                #nn.BatchNorm2d(self.out_channels),
                nn.PReLU())
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=3, padding=1),
                #nn.BatchNorm2d(self.out_channels),
                nn.PReLU()
            )
            '''''''''
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=3, padding=1),
                #nn.BatchNorm2d(self.out_channels),
                nn.PReLU()
            )
            '''

        elif self.model_mode is "CIFAR100":
            print('change')

    def forward(self, x):
        if self.model_mode is "CIFAR10":
            block_1 = self.conv1(x)
            #block_2 = self.Max_pooling_1(block_1)
            block_3 = self.SW(block_1)
            #block_4 = self.Up_1(block_3)
            block_5 = self.conv2(block_3)
            #block_6 = self.conv3(block_5)

            out = block_5
        elif self.model_mode is "CIFAR100":
            print('change')
            '''''''''
            out = self.CIFAR100_conv1(x)
            out = self.CIFAR100_conv2(out)
            out = self.CIFAR100_conv3(out)
            out = self.CIFAR100_conv4(out)
            out = self.CIFAR100_classifier(out)
        elif self.model_mode is "SMALL_REGIME":
            out = self.SMALL_conv1(x)
            out = self.SMALL_conv2(out)
            out = self.SMALL_conv3(out)
            out = self.SMALL_conv4(out)
            out = self.SMALL_conv5(out)
            out = self.SMALL_classifier(out)
        elif self.model_mode is "REGULAR_REGIME":
            out = self.REGULAR_conv1(x)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)
        '''''
        # global average pooling
        #batch_size, channels, height, width = out.size()
        '''''''''
        out = F.avg_pool2d(out, kernel_size=[height, width])
        # out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        #print(self.p)
        '''''


        return out


