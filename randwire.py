import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from graph import RandomGraph
from dictionary import resolution_dict
from dictionary import in_channel_dict
from dictionary import out_channel_dict


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


#############################操作######################################
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                              bias=bias)
        self.prelu = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.conv_2(x)
        x = self.pointwise(x)
        return x


########畳み込み#########
class Unit_conv(nn.Module):
    def __init__(self, node_number, in_channels, out_channels):
        super(Unit_conv, self).__init__()
        self.dropout_rate = 0.2
        self.node = node_number
        self.in_channels = in_channel_dict.get(str(self.node))
        self.out_channels = out_channel_dict.get(str(self.node))
        self.unit = nn.Sequential(

            # SeparableConv2d(self.in_channels, self.out_channels, stride=1),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(self.out_channels),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.unit(x)


##########################
######サンプリング########
class Sampling(nn.Module):
    def __init__(self, node_number, in_channels, out_channels, stride=1):
        super(Sampling, self).__init__()
        self.unit = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.unit(x)


#########################
###アップサンプリング###
class UpSampling(nn.Module):
    def __init__(self, node_number, in_channels, out_channels, stride=1):
        super(UpSampling, self).__init__()
        self.unit = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        return self.unit(x)


#########################
######Self_Attention######

#########################
###########################操作完了###############################

class Node(nn.Module):
    def __init__(self, node, in_degree, in_channels, out_channels, node_num, stride=1):
        super(Node, self).__init__()
        self.in_degree = in_degree
        self.node_number = node
        self.node_total = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        ####SA####
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channel_dict.get(str(self.node_number)),
                      in_channel_dict.get(str(self.node_number)), kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel_dict.get(str(self.node_number)))
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channel_dict.get(str(self.node_number)),
                      in_channel_dict.get(str(self.node_number)), kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel_dict.get(str(self.node_number)))
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channel_dict.get(str(self.node_number)), 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        ##########
        ##########操作判断############
        if (self.node_number <= self.node_total / 2) and (self.node_number % 2 == 0):
            self.unit = Sampling(self.node_number, in_channels, out_channels, stride=stride)
        elif (self.node_number > self.node_total / 2) and (self.node_number % 2 == 0):
            self.unit = UpSampling(self.node_number, in_channels, out_channels, stride=stride)
        else:
            self.unit = Unit_conv(self.node_number, in_channels, out_channels)
        print(self.in_degree)
        ##############################
        ##########正規化##############
        self.normalize = nn.ModuleList()
        for i in range(len(self.in_degree)):
            self.normalize.append(
                nn.Conv2d(out_channel_dict.get(str(self.in_degree[i])), in_channel_dict.get(str(self.node_number)), 1,
                          1, 0, 1, 1, bias=True))
        self.normalize.append(nn.Conv2d(in_channel_dict.get(str(self.node_number)) * len(self.in_degree),
                                        in_channel_dict.get(str(self.node_number)), 1, 1, 0, 1, 1, bias=True))
        ###############################

    def forward(self, *input):
        x = F.interpolate(input[0],
                          size=[resolution_dict.get(str(self.node_number)), resolution_dict.get(str(self.node_number))])
        x = self.normalize[0](x)
        x_standard = x
        if len(self.in_degree) > 1:
            for index in range(1, len(input)):
                y = F.interpolate(input[index], size=[resolution_dict.get(str(self.node_number)),
                                                      resolution_dict.get(str(self.node_number))])
                y = self.normalize[index](y)
                #'''
                g1 = self.W_g(x_standard)
                x1 = self.W_x(y)
                psi = self.relu(g1 + x1)
                psi = self.psi(psi)
                y = y*psi
                #'''
                x = torch.cat([x, y], dim=1)
        x = self.normalize[-1](x)
        out = self.unit(x)

        return out


#############初期化##############
class Node_initial(nn.Module):
    def __init__(self):
        super(Node_initial, self).__init__()

    def forward(self, x):
        return (x)


################################
class RandWire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, is_train, batch_size, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.name = name
        self.batch_size = batch_size
        self.zero_fill = torch.full([self.batch_size, 16, 16, 16], 0).cuda()

        # get graph nodes and in edges
        graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        if self.is_train is True:
            print("is_train: True")
            graph = graph_node.make_graph()
            self.nodes, self.in_edges = graph_node.get_graph_info(graph)
            graph_node.save_random_graph(graph, name)
        else:
            graph = graph_node.load_random_graph(name)
            self.nodes, self.in_edges = graph_node.get_graph_info(graph)

        # define input Node
        self.module_list = nn.ModuleList([Node_initial()])
        # define the rest Node
        self.module_list.extend(
            [Node(node, self.in_edges[node], self.out_channels, self.out_channels, self.node_num) for node in self.nodes
             if node > 0])

    def forward(self, x):
        memory = {}
        # start vertex
        out = self.module_list[0].forward(x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes)):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            elif len(self.in_edges[node]) == 1 and self.in_edges[node][0] == 0 and node > 1:
                y = memory[self.in_edges[node][0]]
                out = self.module_list[node].forward(y.fill_(0))
                # print(node)
                # print(out)

            else:
                out = self.module_list[node].forward(memory[self.in_edges[node][0]])
            memory[node] = out

        out = memory[self.node_num + 1]

        return out
