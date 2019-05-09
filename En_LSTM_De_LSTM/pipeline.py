import os
from torch.utils.data import Dataset
from torch import from_numpy as fn
import pickle
import torch
import sys
import numpy as np

def read_flare_LSTM_data(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data

class TrainDataset(Dataset):
    def __init__(self, root, input_len, output_len):
        super(TrainDataset, self).__init__()
        self.root = root
        self.list_path = os.listdir(root)
        self.input_len = input_len
        self.output_len = output_len

        XX = read_flare_LSTM_data(self.root + 'Xclass_LSTM_model_data_log_ver2_tillPeak.pickle')
        MM = read_flare_LSTM_data(self.root + 'Mclass_LSTM_model_data_log_ver2_tillPeak.pickle')

        Train_input = np.append(XX['Train_input_set'], MM['Train_input_set'], axis=0)
        Train_input = Train_input[:, -self.input_len:]
        Train_input = Train_input.reshape(len(Train_input), self.input_len, 1)

        Train_target = np.append(XX['Train_output_set'], MM['Train_output_set'], axis=0)
        Train_target = Train_target[:, 0: self.output_len]
        Train_target = Train_target.reshape(len(Train_target), self.output_len)

        maxval = np.amax((np.amax(Train_input),
                          np.amax(Train_target)))

        minval = np.amin((np.amin(Train_input),
                          np.amin(Train_target)))

        self.len = len(Train_input)
        self.maxval = maxval
        self.minval = minval

        def Norm(x, maxval, minval):
            return (((x - minval) / (maxval - minval)) - 0.5) / 0.5

        self.Train_input = fn(Norm(Train_input, maxval, minval)).cuda()
        self.Train_input = self.Train_input.type(torch.cuda.FloatTensor)

        self.Train_target = fn(Norm(Train_target, maxval, minval)).cuda()
        self.Train_target = self.Train_target.type(torch.cuda.FloatTensor)

    def __getitem__(self, index):

        Train_input = self.Train_input[index]
        Train_target = self.Train_target[index]

        return Train_input, Train_target

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, root, input_len, output_len):
        super(TestDataset, self).__init__()
        self.root = root
        self.list_path = os.listdir(root)
        self.input_len = input_len
        self.output_len = output_len

        XX = read_flare_LSTM_data(self.root + 'Xclass_LSTM_model_data_log_ver2_tillPeak.pickle')
        MM = read_flare_LSTM_data(self.root + 'Mclass_LSTM_model_data_log_ver2_tillPeak.pickle')

        Train_input = np.append(XX['Train_input_set'], MM['Train_input_set'], axis=0)
        Train_input = Train_input[:, -self.input_len:]

        Train_target = np.append(XX['Train_output_set'], MM['Train_output_set'], axis=0)
        Train_target = Train_target[:, 0: self.output_len]

        maxval = np.amax((np.amax(Train_input),
                          np.amax(Train_target)))

        minval = np.amin((np.amin(Train_input),
                          np.amin(Train_target)))

        Test_input = np.append(XX['Test_input_set'], MM['Test_input_set'], axis=0)
        Test_input = Test_input[:, -self.input_len:]
        Test_input = Test_input.reshape(len(Test_input), self.input_len, 1)

        Test_target = np.append(XX['Test_output_set'], MM['Test_output_set'], axis=0)
        Test_target = Test_target[:, 0:self.output_len]
        Test_target = Test_target.reshape(len(Test_target), self.output_len)

        self.len = len(Test_input)
        self.maxval = maxval
        self.minval = minval

        def Norm(x, maxval, minval):
            return (((x - minval) / (maxval - minval)) - 0.5) / 0.5

        self.Test_input = fn(Norm(Test_input, maxval, minval)).cuda()
        self.Test_input = self.Test_input.type(torch.cuda.FloatTensor)

        self.Test_target = fn(Norm(Test_target, maxval, minval)).cuda()
        self.Test_target = self.Test_target.type(torch.cuda.FloatTensor)

    def __getitem__(self, index):

        Test_input = self.Test_input[index]
        Test_target = self.Test_target[index]

        return Test_input, Test_target

    def __len__(self):
        return self.len

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    CTLG_root = 'c:\\Projects\\FLARE\\'

    data = TestDataset(root=CTLG_root, input_len=30, output_len=30)
    data_loader = DataLoader(dataset=data, batch_size=2)
    # print(data.Test_target[0:2])
    # print('aaa')
    i = 0
    for input, target in data_loader:
        i += 1
        print(target)

        break

        if i >5:
            break
