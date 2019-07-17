import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class MajorMinor_LSTMmodel(nn.Module):
    def __init__(self, feature_size, hidden_size, label_length, num_lstm_layer, num_linear_layer, batch_size, TOTAL_NUM, drop_frac = 0.0):
        super(MajorMinor_LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.label_length = label_length
        self.num_lstm_layer = num_lstm_layer
        self.num_linear_layer = num_linear_layer
        self.batch_size = batch_size
        self.batch_size_fin = TOTAL_NUM % batch_size
        self.drop_frac = drop_frac

        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layer,
                            batch_first=True)

        grow_ratio = 1
        for i in range(self.num_linear_layer):
            if i == 0:
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=self.hidden_size,
                                                               out_features=grow_ratio * self.hidden_size))
            elif i < (self.num_linear_layer-1):
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=grow_ratio * self.hidden_size//(2**(i-1)),
                                                               out_features=grow_ratio * self.hidden_size//(2**(i+0))))
            elif i == (self.num_linear_layer-1):
                setattr(self, 'linear_{}'.format(i), nn.Linear(in_features=grow_ratio * self.hidden_size//(2**(i-1)),
                                                               out_features=self.label_length))
            else:
                raise NotImplementedError("Somethings are wrong at __init__ in Model")


    def forward(self, input):
        x, (hidden, cell) = self.lstm(input)
        hidden = hidden[-1]
        for i in range(self.num_linear_layer):
            if i < (self.num_linear_layer-1):
                hidden = getattr(self, 'linear_{}'.format(i))(hidden)
                hidden = F.dropout(F.relu(hidden), p=self.drop_frac)
            elif i == (self.num_linear_layer - 1):
                hidden = getattr(self, 'linear_{}'.format(i))(hidden)
            else:
                NotImplementedError("Somethings are wrong at forward in Model")
        return hidden
