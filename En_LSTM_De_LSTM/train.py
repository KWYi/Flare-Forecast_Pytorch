import os, sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime

from tqdm import trange
from torch.utils.data import DataLoader
from En_LSTM_De_LSTM.pipeline import TrainDataset, TestDataset
from En_LSTM_De_LSTM.models import Encoder, Decoder

def save_model(Save_path, model_name, epoch, name=[], layer_info=' ', additional_info=' '):
    if Save_path[-2:] != '\\':
        Save_path = Save_path + '\\'
    model_name = model_name
    if (len(name) > 0 and type(name) == str):
        model_name = model_name + '_' + name
    if not os.path.exists(Save_path + model_name):
        os.mkdir(Save_path + model_name)
    if not os.path.exists(Save_path + model_name + '\\ep' + str(epoch)):
        os.mkdir(Save_path + model_name + '\\ep' + str(epoch))

    torch.save(E.state_dict(), Save_path + model_name + '\\ep' + str(epoch) + '\\Encoder' + model_name + '.h5')
    torch.save(D.state_dict(), Save_path + model_name + '\\ep' + str(epoch) + '\\Decoder' + model_name + '.h5')
    Summary = open(Save_path + model_name + '\\ep' + str(epoch) + '\\Summary_' + model_name + '.txt', 'w')
    Summary.write(str(E))
    Summary.write('\n')
    Summary.write('\n')
    Summary.write(str(D))
    Summary.write('\n')
    Summary.write('\n')
    Summary.write('Input_Sequence_Length: {}'.format(input_len))
    Summary.write('\n')
    Summary.write('\n')
    Summary.write('Output_Sequence_Length: {}'.format(output_len))
    Summary.close()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

CTLG_root = 'c:\\Projects\\FLARE\\'

batch_size = 400
input_len = 30
output_len = 30
Train_data = TrainDataset(root=CTLG_root, input_len=input_len, output_len=output_len)
Train_data_loader = DataLoader(dataset=Train_data, batch_size=batch_size, shuffle=False)
# Test_data = TestDataset(root=CTLG_root, input_len=30, output_len=30)
# Test_data_loader = DataLoader(dataset=Test_data, batch_size=batch_size, shuffle=False)

TRAIN_TOTAL_NUM = Train_data.__len__()
E = Encoder(1, 50 ,2, batch_size, TRAIN_TOTAL_NUM).to(DEVICE)
D = Decoder(1, 50, 2, 1).to(DEVICE)

loss = nn.MSELoss().to(DEVICE)
E_optim = torch.optim.Adam(E.parameters(), lr=0.001, betas=[0.9, 0.99])
D_optim = torch.optim.Adam(D.parameters(), lr=0.001, betas=[0.9, 0.99])

total_step = 0
list_loss = []

Save_path = 'c:\\Users\\YKW\\PycharmProjects\\FlarePrediction\\En_LSTM_De_LSTM\\Results\\'
if Save_path[-2:] != '\\':
    Save_path = Save_path+'\\'
filelist = os.listdir(Save_path)
dt = datetime.datetime.now()
dt = dt.strftime("%Y%m%d")
model_number = 1
for i in filelist:
    if i[0:8] == dt:
        model_number += 1
model_name = dt+"_model"+str(model_number)

EPOCHS = 200
save_term = 10
for epoch in trange(EPOCHS):
    epoch +=1
    for Train_input, Train_target in Train_data_loader:
        total_step +=1
        Train_input, Train_target = Train_input.to(DEVICE), Train_target.to(DEVICE)

        Encoder_result, Encoder_hidden = E(Train_input)

        Decoder_input = Train_input[:,-1,:].view(len(Encoder_result),1,1)
        Decoder_hidden = Encoder_hidden
        Decoder_result = torch.zeros(len(Encoder_result),output_len).to(DEVICE)
        # Save model results in array batch_size*output_len
        for di in range(output_len):
            Decoder_input, Decoder_hidden = D(Decoder_input, Decoder_hidden)
            Decoder_result[:,di] = Decoder_input.view(len(Encoder_result))

        l = loss(Decoder_result, Train_target)

        list_loss.append(l.detach().item())

        E_optim.zero_grad()
        D_optim.zero_grad()

        l.backward()

        E_optim.step()
        D_optim.step()
    if epoch%save_term ==0:
        save_model(Save_path, model_name, epoch)

plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams['lines.linewidth'] = 1

plt.plot(range(len(list_loss)), list_loss)
plt.savefig(Save_path+model_name+'\\'+model_name+'_loss_graph.png')
