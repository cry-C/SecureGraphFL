import torch
import torch.nn as nn
import numpy as np

def evaluate_accuracy(global_model, global_model_err, global_model_yuan, err_val_iter, val_loader, adj_matrix, zscore, err_zscore, device):
    global_model.eval()
    global_model_err.eval()
    global_model_yuan.eval()
    loss = nn.L1Loss()
    mae_val = []
    valid_losses = []
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for (inputs_err, labels_err), (inputs, labels) in zip(err_val_iter, val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_err, labels_err = inputs_err.to(device), labels_err.to(device)

            outputs_yuan = global_model_yuan(inputs, adj_matrix, device)
            outputs_err = global_model_err(inputs_err, adj_matrix, device)

            outputs_yuan_cpu = zscore.inverse_transform(outputs_yuan.detach().cpu().numpy())
            outputs_yuan_z = torch.from_numpy(outputs_yuan_cpu).to(device)
            outputs_err_cpu = err_zscore.inverse_transform(outputs_err.detach().cpu().numpy())
            outputs_err_z = torch.from_numpy(outputs_err_cpu).to(device)

            outputs = global_model(outputs_err_z, outputs_yuan_z)
            labels_yuan = zscore.inverse_transform(labels.detach().cpu().numpy())
            labels_yuan = torch.from_numpy(labels_yuan).to(device)

            outputs_cpu = outputs.detach().cpu().numpy().reshape(-1)
            labels_cpu = zscore.inverse_transform(labels.detach().cpu().numpy()).reshape(-1)
            d = np.abs(labels_cpu - outputs_cpu)
            mae_val += d.tolist()
            loss3 = loss(outputs, labels_yuan)
            valid_losses.append(loss3.item())
            l_sum += loss3.item() * labels.shape[0]
            n += labels.shape[0]
        mse = l_sum / n
        MAE = np.array(mae_val).mean()

        return torch.tensor(mse), MAE, valid_losses


def evaluate_model(global_model, global_model_err, global_model_yuan, err_test_iter, test_loader, adj_matrix, zscore, err_zscore, device):
    global_model.eval()
    global_model_err.eval()
    global_model_yuan.eval()
    loss = nn.L1Loss()  # torch.nn.MSELoss()
    l_sum, n = 0.0, 0
    valid_losses = []
    with torch.no_grad():
        for (inputs_err, labels_err), (inputs, labels) in zip(err_test_iter, test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_err, labels_err = inputs_err.to(device), labels_err.to(device)

            outputs_yuan = global_model_yuan(inputs, adj_matrix, device)
            outputs_err = global_model_err(inputs_err, adj_matrix, device)

            outputs_yuan_cpu = zscore.inverse_transform(outputs_yuan.detach().cpu().numpy())
            outputs_yuan_z = torch.from_numpy(outputs_yuan_cpu).to(device)
            outputs_err_cpu = err_zscore.inverse_transform(outputs_err.detach().cpu().numpy())
            outputs_err_z = torch.from_numpy(outputs_err_cpu).to(device)

            outputs = global_model(outputs_err_z, outputs_yuan_z)
            labels_yuan = zscore.inverse_transform(labels.detach().cpu().numpy())
            labels_yuan = torch.from_numpy(labels_yuan).to(device)

            outputs_cpu = outputs.detach().cpu().numpy().reshape(-1)
            labels_cpu = zscore.inverse_transform(labels.detach().cpu().numpy()).reshape(-1)

            loss3 = loss(outputs, labels_yuan)
            valid_losses.append(loss3.item())
            l_sum += loss3.item() * labels.shape[0]
            n += labels.shape[0]
        mse = l_sum / n

        return mse, valid_losses

def evaluate_metric(global_model, global_model_err, global_model_yuan, err_test_iter, test_loader, adj_matrix, zscore, err_zscore, device):
    global_model.eval()
    global_model_err.eval()
    global_model_yuan.eval()
    with torch.no_grad():
        mae_test, sum_y, mape, mse = [], [], [], []
        for (inputs_err, labels_err), (inputs, labels) in zip(err_test_iter, test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_err, labels_err = inputs_err.to(device), labels_err.to(device)

            outputs_yuan = global_model_yuan(inputs, adj_matrix, device)
            outputs_err = global_model_err(inputs_err, adj_matrix, device)

            outputs_yuan_cpu = zscore.inverse_transform(outputs_yuan.detach().cpu().numpy())
            outputs_yuan_z = torch.from_numpy(outputs_yuan_cpu).to(device)
            outputs_err_cpu = err_zscore.inverse_transform(outputs_err.detach().cpu().numpy())
            outputs_err_z = torch.from_numpy(outputs_err_cpu).to(device)

            outputs = global_model(outputs_err_z, outputs_yuan_z)
            labels_yuan = zscore.inverse_transform(labels.detach().cpu().numpy())
            labels_yuan = torch.from_numpy(labels_yuan).to(device)

            outputs_cpu = outputs.detach().cpu().numpy().reshape(-1)
            labels_cpu = zscore.inverse_transform(labels.detach().cpu().numpy()).reshape(-1)

            d = np.abs(labels_cpu - outputs_cpu)
            mae_test += d.tolist()

            non_zero_indices = labels_cpu != 0
            mape += (d[non_zero_indices] / labels_cpu[non_zero_indices]).tolist()

            sum_y += labels_cpu.tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae_test).mean()
        MAPE = np.array(mape).mean()*100 if mape else 0
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE