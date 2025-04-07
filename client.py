#client.py
import torch
import time
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
import os

from pytorchtools import EarlyStopping
import myutils

class Client:
    def __init__(self, client_id, train_loader, val_loader, test_loader, err_train_loader, err_val_loader,
                 err_test_loader,adj, device, args, zscore, err_zscore):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.err_train_loader = err_train_loader
        self.err_val_loader = err_val_loader
        self.err_test_loader = err_test_loader
        self.device = device
        self.train_losses = []
        self.valid_losses = []
        self.avg_train_losses = []
        self.avg_valid_losses = []
        self.adj = adj
        self.args = args
        self.zscore = zscore
        self.err_zscore = err_zscore
        self.avg_train_MAE_losses = []
        self.avg_valid_MAE_losses = []

        self.save_folder = './pictures/Client/'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.bestmodel = None
        self.bestmodel_yuan = None
        self.bestmodel_err = None

    def train(self, model, model_yuan, model_err, num_epochs, lr, epochs):
        self.train_losses = []
        self.valid_losses = []
        self.avg_train_losses = []
        self.avg_valid_losses = []
        self.avg_train_MAE_losses = []
        self.avg_valid_MAE_losses = []
        lr = lr
        testLoss_test, MAE_test, MAPE_test, RMSE_test = self.evaluate(model, model_err, model_yuan)
        with open('./res_MAE.txt', 'a') as time_file:
            time_file.write(f"Before Train Client {self.client_id} Epoch {epochs + 1}: TestLoss {testLoss_test:.6f} MAE {MAE_test:.6f} MAPE {MAPE_test:.6f} RMSE {RMSE_test:.6f}\n")
        print(f"Before Train Client {self.client_id} Epoch {epochs + 1}: TestLoss {testLoss_test:.6f} MAE {MAE_test:.6f} MAPE {MAPE_test:.6f} RMSE {RMSE_test:.6f}")

        epochs = epochs
        client_start = time.time()
        optimizer1 = optim.Adam(model_yuan.parameters(), lr=lr)
        optimizer2 = optim.Adam(model_err.parameters(), lr=lr)
        optimizer3 = optim.Adam(model.parameters(), lr=lr)
        criterion1 = nn.HuberLoss(delta=0.25)
        criterion2 = nn.HuberLoss(delta=0.9)
        criterion3 = nn.L1Loss()
        early_stopping = EarlyStopping(patience=7, verbose=True)
        model = model.to(self.device)
        model_yuan = model_yuan.to(self.device)
        model_err = model_err.to(self.device)

        for epoch in range(num_epochs):
            mae = []
            client_epoch_start = time.time()
            model.train()
            model_yuan.train()
            model_err.train()
            running_loss = 0.0
            for (inputs_err, labels_err), (inputs, labels) in zip(self.err_train_loader, self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs_err, labels_err = inputs_err.to(self.device), labels_err.to(self.device)
                outputs_yuan = model_yuan(inputs, self.adj, self.device)
                outputs_err = model_err(inputs_err, self.adj, self.device)

                outputs_yuan_cpu = self.zscore.inverse_transform(outputs_yuan.detach().cpu().numpy())
                outputs_yuan_z = torch.from_numpy(outputs_yuan_cpu).to(self.device)
                outputs_err_cpu = self.err_zscore.inverse_transform(outputs_err.detach().cpu().numpy())
                outputs_err_z = torch.from_numpy(outputs_err_cpu).to(self.device)

                outputs = model(outputs_err_z, outputs_yuan_z)
                labels_yuan = self.zscore.inverse_transform(labels.detach().cpu().numpy())
                labels_yuan = torch.from_numpy(labels_yuan).to(self.device)

                outputs_cpu = outputs.detach().cpu().numpy().reshape(-1)
                labels_cpu = self.zscore.inverse_transform(labels.detach().cpu().numpy()).reshape(-1)

                d = np.abs(labels_cpu - outputs_cpu)
                mae += d.tolist()

                optimizer1.zero_grad()
                loss1 = criterion1(outputs_yuan, labels)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                loss2 = criterion2(outputs_err, labels_err)
                loss2.backward()
                optimizer2.step()

                optimizer3.zero_grad()
                loss3 = criterion3(outputs, labels_yuan)
                loss3.backward()
                optimizer3.step()

                running_loss += loss3.item()
                self.train_losses.append(loss3.item())
            MAE_Train = np.array(mae).mean()
            self.avg_train_MAE_losses.append(MAE_Train)

            Valloss, MAE_Val, valid_losses = self.evaluate_accuracy(model, model_err, model_yuan)
            self.avg_valid_MAE_losses.append(MAE_Val)
            self.valid_losses.append(valid_losses)

            client_epoch_end = time.time()
            client_epoch = client_epoch_end - client_epoch_start
            with open('./FedAvg_cpu_early_time.txt', 'a') as time_file:
                time_file.write(f"Client {self.client_id} Epoch {epoch+1} total time: {client_epoch} seconds\n")
            with open('./res_loss.txt', 'a') as time_file:
                time_file.write(f"Client {self.client_id} Epoch {epoch+1}: Loss {running_loss/len(self.train_loader)}, Val loss: {Valloss}\n")
            print(f"Client {self.client_id} Epoch {epoch+1}: Loss {running_loss/len(self.train_loader)}, Val loss: {Valloss}")

            with open('./res_MAE.txt', 'a') as time_file:
                time_file.write(f"Client {self.client_id} Epoch {epoch + 1}: Train MAE {MAE_Train}, Val MAE: {MAE_Val}\n")
            print(f"Client {self.client_id} Epoch {epoch + 1}: Train MAE {MAE_Train}, Val MAE: {MAE_Val}\n")

            train_loss = np.average(self.train_losses)
            valid_loss = np.average(self.valid_losses)
            self.avg_train_losses.append(train_loss)
            self.avg_valid_losses.append(valid_loss)
            self.train_losses = []
            self.valid_losses = []
            self.bestmodel , self.bestmodel_yuan, self.bestmodel_err = early_stopping(MAE_Val, model, model_yuan, model_err)
            if early_stopping.early_stop:
                print("Client Early stopping")
                break
        testLoss, MAE, MAPE, RMSE = self.evaluate(self.bestmodel, model_err, model_yuan)
        with open('./res_MAE.txt', 'a') as time_file:
            time_file.write(
                f"After Train Client {self.client_id} Epoch {epochs + 1}: TestLoss {testLoss:.6f} MAE {MAE:.6f} MAPE {MAPE:.6f} RMSE {RMSE:.6f}\n")
        print(f"After Train Client {self.client_id} Epoch {epochs + 1}: TestLoss {testLoss:.6f} MAE {MAE:.6f} MAPE {MAPE:.6f} RMSE {RMSE:.6f}")
        client_end = time.time()
        client_time = client_end - client_start
        with open('./FedAvg_cpu_early_time.txt', 'a') as time_file:
            time_file.write(f"Client {self.client_id} total time: {client_time} seconds\n")

        self.avg_train_MAE_losses = np.array(self.avg_train_MAE_losses)
        self.avg_valid_MAE_losses = np.array(self.avg_valid_MAE_losses)
        self.avg_train_losses = np.array(self.avg_train_losses)
        self.avg_valid_losses = np.array(self.avg_valid_losses)

        fig = plt.figure(figsize=(15, 12))
        plt.plot(range(1, len(self.avg_train_MAE_losses) + 1), self.avg_train_MAE_losses, label='Training MAE')
        plt.plot(range(1, len(self.avg_valid_MAE_losses) + 1), self.avg_valid_MAE_losses, label='Validation MAE')
        count_all = np.argmin(self.avg_valid_MAE_losses) + 1
        plt.axvline(count_all, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('MAE')
        plt.ylim(1, 5)
        plt.xlim(0, len(self.avg_train_MAE_losses) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        file_name = f'Client {self.client_id} Epoch {epochs + 1} MAE_plot.png'
        fig.savefig(os.path.join(self.save_folder, file_name), bbox_inches='tight')

        fig = plt.figure(figsize=(15, 12))
        plt.plot(range(1, len(self.avg_train_losses) + 1), self.avg_train_losses, label='Training Loss')
        plt.plot(range(1, len(self.avg_valid_losses) + 1), self.avg_valid_losses, label='Validation Loss')
        minposs = np.argmin(self.avg_valid_losses) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Min Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(1, 5)
        plt.xlim(0, len(self.avg_train_losses) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        file_name = f'Client {self.client_id} Epoch {epochs + 1} loss_plot.png'
        fig.savefig(os.path.join(self.save_folder, file_name), bbox_inches='tight')
        self.bestmodel.cpu()
        self.bestmodel_yuan.cpu()
        self.bestmodel_err.cpu()
        return self.bestmodel.state_dict(), self.bestmodel_yuan.state_dict(), self.bestmodel_err.state_dict(), client_time  # 返回训练后的模型参数

    def evaluate_accuracy(self, model, model_err, model_yuan):
        mse, MAE, valid_losses = myutils.evaluate_accuracy(model, model_err, model_yuan, self.err_val_loader, self.val_loader, self.adj, self.zscore, self.err_zscore, self.device)
        return mse, MAE, valid_losses

    def evaluate(self, model, model_err, model_yuan):
        model = model.to(self.device)
        test_MSE, _ = myutils.evaluate_model(model, model_err, model_yuan, self.err_test_loader, self.test_loader, self.adj, self.zscore, self.err_zscore, self.device)
        test_MAE, test_MAPE, test_RMSE = myutils.evaluate_metric(model, model_err, model_yuan, self.err_test_loader, self.test_loader, self.adj, self.zscore, self.err_zscore, self.device)
        return test_MSE, test_MAE, test_MAPE, test_RMSE