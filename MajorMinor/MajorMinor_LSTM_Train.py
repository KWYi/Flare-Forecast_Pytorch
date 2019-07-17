if __name__ == '__main__':
    import os, sys
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    from tqdm import tqdm, trange
    from torch.utils.data import DataLoader
    from MajorMinor.MajorMinor_pipeline import XrayDataset
    from MajorMinor.MajorMinor_Model1_LSTM import MajorMinor_LSTMmodel
    from MajorMinor.Early_Stopping import EarlyStopping

    Early_Stopping = EarlyStopping(patience=5, verbose=True)

    def save_model(Save_path, model_name, epoch, name=[], layer_info=' ', additional_info=' '):
        model_name = model_name
        if (len(name) > 0 and type(name) == str):
            model_name = model_name + '_' + name
        if not os.path.exists(Save_path + model_name):
            os.mkdir(Save_path + model_name)
        if not os.path.exists(Save_path + model_name + '\\ep' + str(epoch)):
            os.mkdir(Save_path + model_name + '\\ep' + str(epoch))

        torch.save(Model.state_dict(), Save_path + model_name + '\\ep' + str(epoch) + '\\MajorMinor' + model_name + '.h5')
        Summary = open(Save_path + model_name + '\\ep' + str(epoch) + '\\Summary_' + model_name + '.txt', 'w')
        Summary.write(str(Model))
        Summary.write('\n')
        Summary.write('\n')
        Summary.write('Input_Sequence_Length: {}'.format(input_len))
        Summary.close()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    CTLG_root = 'c:\\Projects\\FLARE\\Data_for_model\\'

    output_type = 'binary'
    feature_size = 1
    unit_size = 256
    num_lstm_layer = 2
    num_linear_layer = 2
    drop_ratio = 0.5
    batch_size = 50
    input_len = 30
    num_workers = 3
    EPOCHS = 200
    if output_type == 'binary':
        label_length = 1
    elif output_type == 'class':
        label_length = 4
    else:
        raise NotImplementedError("output_type error")

    Save_path = 'c:\\Users\\YKW\\PycharmProjects\\FlarePrediction_ver2\\MajorMinor\\Results'
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

    Train_data = XrayDataset(mod='Train', path_dir=CTLG_root, input_len=input_len, feature_size=feature_size, output_type=output_type)
    Test_data = XrayDataset(mod='Test', path_dir=CTLG_root, input_len=input_len, feature_size=feature_size, output_type=output_type)
    TRAIN_TOTAL_NUM = len(Train_data)
    TEST_TOTAL_NUM = len(Test_data)
    Train_data_loader = DataLoader(dataset=Train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    Test_data_loader = DataLoader(dataset=Test_data, batch_size=20000, num_workers=num_workers, shuffle=False)

    Model = MajorMinor_LSTMmodel(feature_size=feature_size, hidden_size=unit_size, label_length=label_length,
                                 num_lstm_layer=num_lstm_layer, num_linear_layer=num_linear_layer, batch_size=batch_size,
                                 TOTAL_NUM=TRAIN_TOTAL_NUM, drop_frac=drop_ratio).to(device=DEVICE)

    Model_optim = torch.optim.Adam(Model.parameters(), lr=0.001, betas=[0.9, 0.99])
    if output_type == 'binary':
        loss = nn.BCEWithLogitsLoss().to(device=DEVICE)
    elif output_type == 'class':
        loss = nn.CrossEntropyLoss().to(device=DEVICE)
    else:
        raise NotImplementedError("You entered wrong output_type")

    train_loss_plot = np.array([])
    test_loss_plot = np.array([])

    for epoch in trange(EPOCHS):
        epoch += 1

        train_losses = []
        for Train_input, Train_target in Train_data_loader:

            if output_type == 'binary':
                Train_input, Train_target = Train_input.to(device=DEVICE, dtype=torch.float32), \
                                            Train_target.to(device=DEVICE, dtype=torch.float32)
            elif output_type == 'class':
                Train_input, Train_target = Train_input.to(device=DEVICE, dtype=torch.float32), \
                                            Train_target.to(device=DEVICE, dtype=torch.long)
                Train_target = Train_target.view(-1)
            else:
                raise NotImplementedError("Output_type is wrong")

            Result = Model(Train_input)
            l = loss(Result, Train_target)
            train_losses.append(l.detach().item())

            Model_optim.zero_grad()
            l.backward()
            Model_optim.step()

        Test_losses = []
        for Test_input, Test_target in Test_data_loader:
            if output_type == 'binary':
                Test_input, Test_target = Test_input.to(device=DEVICE, dtype=torch.float32),\
                                          Test_target.to(device=DEVICE, dtype=torch.float32)
            elif output_type == 'class':
                Test_input, Test_target = Test_input.to(device=DEVICE, dtype=torch.float32),\
                                          Test_target.to(device=DEVICE, dtype=torch.long)
                Test_target = Test_target.view(-1)
            else:
                raise NotImplementedError("Output_type is wrong")

            test_Result = Model(Test_input)
            test_l = loss(Test_Result, Test_target)
            test_losses.append(Test_l.detach().item())

        train_loss = np.average(train_losses)
        test_loss = np.average(test_losses)

        train_loss_plot = np.append(train_loss_plot, train_loss)
        test_loss_plot = np.append(test_loss_plot, test_loss)

        # save_model(Save_path, model_name, epoch)
        # if Early_Stopping.early_stop:
        #     save_model(Save_path, model_name, epoch)
        #     print("Stopped at {}epoch!".format(epoch))

    save_model(Save_path, model_name, epoch)
    plt.rcParams["figure.figsize"] = (8,5)
    plt.rcParams['lines.linewidth'] = 1

    plt.plot(range(len(train_loss_plot)), train_loss_plot, label='Train')
    plt.plot(range(len(train_loss_plot)), test_loss_plot, label='Test')
    plt.legend(loc='upper right')
    plt.savefig(Save_path+model_name+'\\'+model_name+'_loss_graph.png')