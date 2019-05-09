import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, layer_num, batch_size, TOTAL_NUM):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.batch_size_fin = TOTAL_NUM % batch_size
        self.state_zero = (nn.Parameter(torch.zeros(layer_num, batch_size, hidden_size)).to(DEVICE),
                           nn.Parameter(torch.zeros(layer_num, batch_size, hidden_size)).to(DEVICE))
        self.state_zero_fin = (nn.Parameter(torch.zeros(layer_num, self.batch_size_fin, hidden_size)).to(DEVICE),
                               nn.Parameter(torch.zeros(layer_num, self.batch_size_fin, hidden_size)).to(DEVICE))

        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.layer_num, batch_first=True)

    def forward(self, input):

        if input.shape[0] == self.batch_size:
            x, hidden = self.lstm(input, self.state_zero)
            self.hidden = hidden
        elif input.shape[0] == self.batch_size_fin:
            x, hidden = self.lstm(input, self.state_zero_fin)
            self.hidden = hidden
        else:
            print(input.shape[0],  self.batch_size, self.batch_size_fin)
            raise ValueError(
                "We got wrong initial batch_size!: expected {} or {}, but got {}".format(self.batch_size,  self.batch_size_fin, input.shape[0]))

        return x, self.hidden

class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, layer_num, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.layer_num = layer_num
        self.output_size = output_size

        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.layer_num, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden):
        x, hidden = self.lstm(input, hidden)
        x = self.dense(x)

        return x, hidden

if __name__ == '__main__':
    None
