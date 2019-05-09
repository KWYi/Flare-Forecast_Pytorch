import numpy as np
import torch
import matplotlib.pyplot as plt
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

plt.rcParams["figure.figsize"] = (15,9)
plt.rcParams['lines.linewidth'] = 6
plt.rcParams.update({'font.size': 12})


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CTLG_root = 'c:\\Projects\\FLARE\\'

batch_size = 2000
input_len = 30
output_len = 30
Test_data = TestDataset(root=CTLG_root, input_len=input_len, output_len=output_len)
Test_data_loader = DataLoader(dataset=Test_data, batch_size=batch_size, shuffle=False)

TEST_TOTAL_NUM = Test_data.__len__()

Model_name = '20190508_model5'
Encoders = []
Decoders = []

OLD_BEST = np.array([0.10598546, 0.13852453, 0.16455523, 0.18673718, 0.20685236,
                     0.22470148, 0.24031378, 0.25412119, 0.26635355, 0.27724988,
                     0.28702851, 0.29584786, 0.30377559, 0.31094392, 0.31742038,
                     0.32330368, 0.32868212, 0.3336029 , 0.33812359, 0.34229802,
                     0.34615127, 0.34973677, 0.35305834, 0.35612333, 0.3589454 ,
                     0.36153522, 0.3638863 , 0.36600662, 0.36791608, 0.36963158])

for ep in range(130, 130+1, 10):
    E = Encoder(1, 60, 2, batch_size, TEST_TOTAL_NUM).to(DEVICE)
    D = Decoder(1, 60, 2, 1).to(DEVICE)

    trained_Encoder = torch.load('c:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Results\\{}\\ep{}\\Encoder{}.h5'\
                                 .format(Model_name,ep,Model_name))
    E.load_state_dict(trained_Encoder)

    trained_Decoder = torch.load('c:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Results\\{}\\ep{}\\Decoder{}.h5'\
                                 .format(Model_name, ep, Model_name))
    D.load_state_dict(trained_Decoder)

    for input, target in Test_data_loader:
        Encoder_result, Encoder_hidden = E(input)
        Decoder_input = input[:, -1, :].view(len(Encoder_result), 1, 1)
        Decoder_hidden = Encoder_hidden
        Decoder_result = torch.zeros(len(Encoder_result), output_len).to(DEVICE)
        for di in range(output_len):
            Decoder_input, Decoder_hidden = D(Decoder_input, Decoder_hidden)
            Decoder_result[:, di] = Decoder_input.detach().view(len(Encoder_result))
    model_rmse = rmse_len(Decoder_result, target)
    # plt.plot(model_rmse, label = 'EPOCH {}, {}'.format(ep,model_rmse[-1]))
plt.plot(OLD_BEST,   label='OLD_BEST // 30min RMSE: {0:3f}'.format(OLD_BEST[-1]))
plt.plot(model_rmse, label='En_De       // 30min RMSE: {0:3f}'.format(model_rmse[-1]))
plt.ylabel("RMSE(W/m^2)", fontsize=20)
plt.xlabel("Forecast Time(min)", fontsize=20)
plt.legend(loc='upper left', fontsize='large')
plt.show()
