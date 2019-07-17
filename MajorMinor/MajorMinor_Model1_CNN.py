import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class MajorMinor_CNNmodel(nn.Module):
    def __init__(self, feature_size, label_length, num_conv_layer, num_linear_layer, kernel_size, padding, stride, drop_frac = 0.0):
        super(MajorMinor_CNNmodel, self).__init__()
        self.feature_size = feature_size
        self.label_length = label_length
        self.num_conv_layer = num_conv_layer
        self.num_linear_layer = num_linear_layer
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.drop_frac = drop_frac

        out_cal = (lambda input_size, padding, kernel_size, stride: np.floor((input_size+2*padding-kernel_size)/stride+1))

        base_ch = 48
        for i in range(self.num_conv_layer):
            if i == 0:
                setattr(self, 'conv1d_{}'.format(i),
                        nn.Conv1d(in_channels=feature_size, out_channels=base_ch, kernel_size=self.kernel_size, padding=self.padding))
                output_size = out_cal(30, self.padding, self.kernel_size, self.stride)
            elif i <= (self.num_conv_layer - 1):
                setattr(self, 'conv1d_{}'.format(i),
                        nn.Conv1d(in_channels=base_ch*i, out_channels=base_ch*(i+1), kernel_size=self.kernel_size, padding=self.padding))
                output_size = out_cal(output_size, self.padding, self.kernel_size, self.stride)
                output_channel = base_ch*(i+1)
            else:
                raise NotImplementedError("Somethings are wrong at Conv1d layer loop in __init__ in Model")

        output_size, output_channel = int(output_size), int(output_channel)

        grow_ratio = 1
        for i in range(self.num_linear_layer):
            if i == 0:
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=output_channel * output_size,
                                                               out_features=grow_ratio * output_size))
            elif i < (self.num_linear_layer-1):
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=grow_ratio * output_size//(2**(i-1)),
                                                              out_features=grow_ratio * output_size//(2**(i+0))))
            elif i == (self.num_linear_layer-1):
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=grow_ratio * output_size//(2**(i-1)),
                                                              out_features=self.label_length))
            else:
                raise NotImplementedError("Somethings are wrong at Linear layer loop in __init__ in Model")

    def forward(self, input):
        input = torch.transpose(input,1,2)
        # Dataloader give model data with (Batch_size, Sequence_length, feature_size) shape, optimized for LSTM,
        # but Conv1d layer need (Batch, channel_size(feature), Data length) shape data. Hence transpose it.
        for i in range(self.num_conv_layer):
            input = getattr(self, 'conv1d_{}'.format(i))(input)
            input = F.dropout(F.relu(input), p=self.drop_frac)

        input = input.view(input.shape[0], -1)

        for i in range(self.num_linear_layer):
            if i < (self.num_linear_layer-1):
                input = getattr(self, 'linear_{}'.format(i))(input)
                input = F.dropout(F.relu(input), p=self.drop_frac)
            elif i == (self.num_linear_layer-1):
                input = getattr(self, 'linear_{}'.format(i))(input)
            else:
                raise NotImplementedError("Somethings are wrong at Linear layer loop in forward in Model")
        return input

# Model to reproduece Gravitiy wave with LIGO model
class GWLIGOmodel(nn.Module):
    def __init__(self, input_len, feature_size, classification_type, batch_norm=False):
        super(GWLIGOmodel, self).__init__()
        norm = nn.BatchNorm1d
        act = nn.ReLU(inplace=True)

        layer = []

        layer += [nn.Linear(in_features=input_len, out_features=8192, bias=False),
                  nn.Conv1d(in_channels=feature_size, out_channels=64, kernel_size=16, stride=1, dilation=1),
                  nn.MaxPool1d(kernel_size=4, stride=4)]
        layer += [nn.BatchNorm1d(64)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        layer += [nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=1, dilation=2),
                  nn.MaxPool1d(kernel_size=4, stride=4)]
        layer += [nn.BatchNorm1d(128)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        layer += [nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=1, dilation=2),
                  nn.MaxPool1d(kernel_size=4, stride=4)]
        layer += [nn.BatchNorm1d(256)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        layer += [nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, stride=1, dilation=2),
                  nn.MaxPool1d(kernel_size=4, stride=4)]
        layer += [nn.BatchNorm1d(512)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        layer += [View(-1),
                  nn.Linear(7168, 128)]
        layer += [nn.BatchNorm1d(128)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        layer += [nn.Linear(128, 64)]
        layer += [nn.BatchNorm1d(64)] if batch_norm else []
        layer += [nn.ReLU(inplace=False)]

        if classification_type == 'CLE':
            layer += [nn.Linear(64, 2)]
        elif classification_type == 'BCE':
            layer += [nn.Linear(64, 1)]
        else:
            raise NotImplementedError("You have to define 'classfication_type' as 'CLE' or 'BCE'")

        self.network = nn.Sequential(*layer).to(device=DEVICE)

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        # Dataloader give model data with (Batch_size, Sequence_length, feature_size) shape, optimized for LSTM,
        # but Conv1d layer need (Batch, channel_size(feature), Data length) shape data. Hence transpose it.
        input = input.to(device=DEVICE)
        return self.network(input)

class View(nn.Module):
    def __init__(self, *shape):  # *의미: *가 붙으면 tuple이 됨.
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

if __name__ =='__main__':
    Model1 = GWLIGOmodel(input_len=30, feature_size=1, classification_type='CLE', extension_type='conv', batch_norm=True)