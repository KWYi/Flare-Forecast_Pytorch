import os
from torch.utils.data import Dataset
from torch import from_numpy
import pickle
import sys
import numpy as np

def read_flare_LSTM_data(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data

class XrayDataset(Dataset):
    def __init__(self, mod, path_dir, input_len, feature_size, output_type='binary'):
        super(XrayDataset, self).__init__()
        self.path_dir = path_dir
        self.list_path = os.listdir(path_dir)
        self.input_len = input_len
        self.feature_size = feature_size
        # self.output_len = output_len

        if output_type == 'binary':
            class2idx = {'B': 0, 'C': 0, 'M': 1, 'X': 1}
        elif output_type == 'class':
            class2idx = {'B': 0, 'C': 1, 'M': 2, 'X': 3}
        else:
            raise NotImplementedError("You have to give model output_type='binary' or 'class'")
        num_class = np.array([0, 0, 0, 0])

        path_dir = 'C:\\Projects\\FLARE\\Data_for_model\\'
        file_list = os.listdir(path_dir)
        file_list.sort()
        Train_file_list = []
        for GOES_seq in file_list:
            if 'Standard_End' in GOES_seq:
                Train_file_list.append(GOES_seq)

        input_list = []
        target_list = []
        for GOES_seq in Train_file_list:
            flare_class = GOES_seq[0]
            seq = read_flare_LSTM_data(self.path_dir + GOES_seq)
            if mod == 'Train':
                seq_input = seq['xl_Train_input_set']
            elif mod == 'Test':
                seq_input = seq['xl_Test_input_set']
            else:
                raise NotImplementedError("You have to enter mod as 'Train' or 'Test'")
            temp_seq_input = []
            for Event_seq in seq_input:
                temp_seq_input.append(Event_seq[0])

            target_label = class2idx[flare_class]

            temp_seq_target = [target_label] * len(temp_seq_input)

            input_list  = input_list + temp_seq_input
            target_list = target_list + temp_seq_target

            num_class[int(class2idx[flare_class])] += len(temp_seq_input)

        Train_input = np.zeros((1, input_len))
        for i in range(len(input_list)):
            Train_input = np.append(Train_input, np.array(input_list[i])[-self.input_len:].reshape(1, input_len), axis=0)
        Train_input = np.delete(Train_input, 0, 0)
        Train_input = Train_input.reshape(len(Train_input), self.input_len, 1)

        maxval = np.amax(Train_input)
        minval = np.amin(Train_input)

        self.len = len(Train_input)
        self.maxval = maxval
        self.minval = minval

        def Norm(x, maxval, minval):
            return (((x - minval) / (maxval - minval)) - 0.5) / 0.5

        if feature_size == 1:
            self.Train_input = from_numpy(Norm(Train_input, maxval, minval))
        elif feature_size ==2:
            self.Train_input = from_numpy(
                np.append(
                    Norm(Train_input, maxval, minval),
                    np.append(
                        np.zeros(self.len).reshape(self.len, 1, 1),
                        np.diff(Norm(Train_input, maxval, minval), n=1, axis=1),
                        axis=1),
                    axis=-1)
            )
        else:
            raise NotImplementedError("You have to set feature size 1 or 2. More value setting will be ready soon")
        target_list = np.array(target_list)
        self.Train_target = from_numpy(target_list.reshape(target_list.shape[0], -1))
        # print(num_class)

    def __getitem__(self, index):
        Train_input = self.Train_input[index]
        Train_target = self.Train_target[index]
        return Train_input, Train_target

    def __len__(self):
        return self.len

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    CTLG_root = 'C:\\Projects\\FLARE\\Data_for_model\\'

    Traindata = TrainDataset(path_dir=CTLG_root, input_len=30, feature_size=1, output_type='binary')
    print(Traindata.Train_input.shape)
    print(Traindata.Train_target.shape)
    Testdata = TestDataset(path_dir=CTLG_root, input_len=30, feature_size=1, output_type='binary')
    print(Testdata.Test_input.shape)
    print(Testdata.Test_target.shape)
