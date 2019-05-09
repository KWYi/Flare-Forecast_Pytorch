import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import trange
from torch.utils.data import DataLoader
from En_LSTM_De_LSTM.pipeline import TestDataset
from En_LSTM_De_LSTM.models import Encoder, Decoder

def de_Norm(data, maxval, minval):
    return ((data/2.)+0.5) * (maxval - minval) + minval

def rmse_len(results, targets):
    results = results.to('cpu')
    targets = targets.to('cpu')
    results, targets = de_Norm(np.array(results),Test_data.maxval, Test_data.minval),\
                       de_Norm(np.array(targets),Test_data.maxval, Test_data.minval)
    SE = (results-targets)**2
    RMSE = np.array([])
    for time in range(SE.shape[-1]):
        RMSE = np.append(RMSE, np.sqrt(np.mean(SE[:,0:time+1])))
    return RMSE

with open('C:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\OLD_BEST_ver2_tillPeak.pickle', 'rb') as f:
    OLD_BEST = pickle.load(f) # 단 한줄씩 읽어옴

plt.rcParams["figure.figsize"] = (15,9)
plt.rcParams['lines.linewidth'] = 6
plt.rcParams.update({'font.size': 12})

MODELS = [
    ['20190508_model5', '130'],
]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CTLG_root = 'c:\\Projects\\FLARE\\'

input_len = 30
output_len = 30
Test_data = TestDataset(root=CTLG_root, input_len=input_len, output_len=output_len)
batch_size = len(Test_data)
Test_data_loader = DataLoader(dataset=Test_data, batch_size=batch_size, shuffle=False)
TEST_TOTAL_NUM = Test_data.__len__()

tail_len = 30
tail_len = min(input_len,tail_len)
x1 = np.linspace(0, tail_len+output_len-1, num=tail_len+output_len)
pred_x = np.linspace(tail_len-1, tail_len-1+output_len, num=tail_len+1)
savelocation = 'C:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Graphs_result\\'
for MODEL in MODELS:
    E = Encoder(1, 60, 2, batch_size, TEST_TOTAL_NUM).to(DEVICE)
    D = Decoder(1, 60, 2, 1).to(DEVICE)
    trained_Encoder = torch.load('c:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Results\\{0}\\ep{1}\\Encoder{0}.h5'\
                                 .format(MODEL[0],MODEL[1]))
    E.load_state_dict(trained_Encoder)
    trained_Decoder = torch.load('c:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Results\\{0}\\ep{1}\\Decoder{0}.h5'\
                                 .format(MODEL[0],MODEL[1]))
    D.load_state_dict(trained_Decoder)

    for input, target in Test_data_loader:
        Encoder_result, Encoder_hidden = E(input)
        Decoder_input = input[:, -1, :].view(len(Encoder_result), 1, 1)
        Decoder_hidden = Encoder_hidden
        Decoder_result = torch.zeros(len(Encoder_result), output_len).to(DEVICE)
        for di in range(output_len):
            Decoder_input, Decoder_hidden = D(Decoder_input, Decoder_hidden)
            Decoder_result[:, di] = Decoder_input.detach().view(len(Encoder_result))

        input = input.view(input.shape[0], -1).to('cpu')
        input = np.array(input)
        input = de_Norm(input, Test_data.maxval, Test_data.minval)
        target = target.view(target.shape[0], -1).to('cpu')
        target = np.array(target)
        target = de_Norm(target, Test_data.maxval, Test_data.minval)
        Decoder_result = Decoder_result.view(Decoder_result.shape[0], -1).to('cpu')
        Decoder_result = np.array(Decoder_result)
        Decoder_result = de_Norm(Decoder_result, Test_data.maxval, Test_data.minval)


        for i in trange(len(Decoder_result)):

            root = input[i, -tail_len:]
            observe = target[i, :]
            predict = Decoder_result[i, :]
            predict = np.append(root[-1], predict)

            plt.rcParams["figure.figsize"] = (12, 6)
            plt.rcParams['lines.linewidth'] = 3
            plt.rcParams.update({'font.size': 15})

            log_plot = plt.plot
            log_plot(x1, np.concatenate((root,observe)), color = 'black', linestyle='-', label='Observation')
            log_plot(pred_x, OLD_BEST[i], linestyle='--', label='OLD_BEST')
            log_plot(pred_x, predict, linestyle='--', label='En_De')
            plt.axvline(x=tail_len-1, color = 'black', linestyle='--', linewidth=3)
            # log_plot.set_ylim([-6.5, -2.5])
            plt.ylabel("Soft X-ray flux(W/m^2)(Logarithm)")
            plt.xlabel("Time(min)")
            plt.legend(loc='upper left')
            plt.savefig(savelocation+str(i)+'.png')
            plt.clf()
