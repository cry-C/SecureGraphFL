# service.py
import random
import torch
import time
from torch import nn
from EarlyStopping import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import argparse
import torch.utils as utils
import copy
import os

from AC import ACLearner
from client import Client
from model import STGCN, GCN
from lodadata import data_preparate,load_sparse_adjacency_matrix
import myutils
random.seed(a=10)
def get_parameters():
    parser = argparse.ArgumentParser(description='FLSTGCN')
    parser.add_argument('--dataset', type=str, default='./data/metr-la',
                        choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3, choices=[3, 1],
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_clients', type=int, default=9, choices=[9, 6])
    args = parser.parse_args()
    return args

class FederatedLearningService:
    def __init__(self, args):
        self.args = args
        device = torch.device('cuda')
        self.device = device
        print(f"Using device: {self.device}")
        self.clients = []
        self.train_loader, self.val_loader, self.test_loader, self.test, self.zscore, self.zscore_client, \
        self.err_train_loader, self.err_val_loader, self.err_test_loader, self.err_test, self.err_zscore, \
        self.err_zscore_client, self.client_n_vertices = data_preparate(args.dataset, args.n_his,args.n_pred,args.num_clients,args.batch_size, self.device)
        self.globel_G, self.client_graph, self.n_vertex = load_sparse_adjacency_matrix(self.args.dataset, self.args.num_clients)

        self.global_model = STGCN(self.n_vertex).to(self.device)
        self.global_model_yuan = GCN(self.args.n_his).to(self.device)
        self.global_model_err = GCN(self.args.n_his - 1).to(self.device)

        self.valid_losses = []
        self.avg_valid_losses = []
        self.avg_valid_MAE_losses = []
        self.avg_valid_client_MAE = []
        self.num_epochs = args.epochs
        self.lr = 0.001
        self.num_clients = args.num_clients
        for i in range(args.num_clients):
            client_train_loader = self.train_loader[i]
            client_val_loader = self.val_loader[i]
            client_test_loader = self.test_loader[i]
            err_client_train_loader = self.err_train_loader[i]
            err_client_val_loader = self.err_val_loader[i]
            err_client_test_loader = self.err_test_loader[i]
            client_adj = self.client_graph[i]
            client = Client(i, client_train_loader, client_val_loader, client_test_loader, err_client_train_loader,
                            err_client_val_loader, err_client_test_loader, client_adj, self.device, args, self.zscore_client[i], self.err_zscore_client[i])
            self.clients.append(client)

        self.AC = ACLearner(input_channels = 1, action_size = self.num_clients)
        self.selection_threshold = 0.5
        self.save_folder = './pictures/AC/'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def train(self):
        early_stopping = EarlyStopping(patience=7, verbose=True)
        for epoch in range(50):
            pool = mp.Pool(processes=self.num_clients)
            epoch_time_start = time.time()
            client_weights = []
            client_yuan_weights = []
            client_err_weights = []
            client_res = []
            client_time = []

            for i in range(self.num_clients):
                global_weights = self.global_model.weight.data
                split_weights = torch.split(global_weights, self.client_n_vertices, dim=1)
                global_bias = self.global_model.bias.data
                split_bias = torch.split(global_bias, self.client_n_vertices, dim=1)
                client_model = STGCN(self.client_n_vertices[i]).to(self.device)
                client_model.weight.data = split_weights[i]
                client_model.bias.data = split_bias[i]

                result = pool.apply_async(self.clients[i].train, args=(client_model, self.global_model_yuan,
                                                                       self.global_model_err, self.num_epochs, self.lr, epoch))
                client_res.append(result)
            pool.close()
            pool.join()
            countall = 0

            for result in client_res:
                model_weights, model_yuan_weights, model_err_weights, c_time = result.get()
                if countall < 2:
                    model_weights = STGCN(self.client_n_vertices[countall]).state_dict()
                    model_yuan_weights = GCN(self.args.n_his).state_dict()
                    model_err_weights = GCN(self.args.n_his - 1).state_dict()
                client_weights.append(model_weights)
                client_yuan_weights.append(model_yuan_weights)
                client_err_weights.append(model_err_weights)
                client_time.append(c_time)
                countall += 1

            all_weights = []
            all_biases = []
            for state_dict in client_weights:
                weight = state_dict['weight']
                bias = state_dict['bias']
                all_weights.append(weight)
                all_biases.append(bias)
            concatenated_weights = torch.cat(all_weights, dim=1)
            concatenated_biases = torch.cat(all_biases, dim=1)
            self.global_model.weight.data = concatenated_weights
            self.global_model.bias.data = concatenated_biases

            client_maes = []
            client_losses = []
            for i in range(self.num_clients):
                client_model_yuan = GCN(self.args.n_his).to(self.device)
                client_model_err = GCN(self.args.n_his - 1).to(self.device)
                client_model_yuan.load_state_dict(client_yuan_weights[i])
                client_model_err.load_state_dict(client_err_weights[i])

                client_loss, client_mae, _, _ = self.evaluate(self.global_model, client_model_err, client_model_yuan)
                client_losses.append(client_loss)
                client_maes.append(client_mae)
            state_client = []
            for i in range(self.args.num_clients):
                temp = []
                temp.append(client_maes[i])
                temp.append(client_time[i])
                state_client.append(temp)

            state = torch.FloatTensor(state_client)
            state = state.cpu()
            state = state.T
            state = state.unsqueeze(0)
            self.avg_valid_client_MAE = []
            for i in range(100):
                action = self.AC.select_action(state)
                actions = (action >= self.selection_threshold).float().cpu().numpy().astype(int)  # .to(torch.int)

                with open('./DQN_state.txt', 'a') as f:
                    f.write(f"Epoch: {epoch + 1}\n")
                    f.write(f"State MAE: {str(client_maes)}\n")
                    f.write(f"State Loss: {str(client_losses)}\n")
                    f.write(f"State TIME: {str(client_time)}\n\n")
                with open('./DQN_action.txt', 'a') as f:
                    f.write(f"Epoch: {epoch + 1}\n")
                    f.write(f"Action: {actions.tolist()}\n")

                action_np = action.detach().cpu().numpy()
                with open('./res_action.txt', 'a') as time_file:
                    time_file.write(f"Epoch {epoch + 1}: Action {action_np}\n")

                choose_client = []
                choose_yuan_client = []
                choose_err_client = []
                for i in range(len(actions)):
                    if actions[i] == 1:
                        choose_client.append(client_weights[i])
                        choose_yuan_client.append(client_yuan_weights[i])
                        choose_err_client.append(client_err_weights[i])

                len_choose = len(choose_client)
                with open('./res_action.txt', 'a') as time_file:
                    time_file.write(f"Service Epoch {epoch + 1}: len_choose {len_choose}\n\n")

                avg_yuan_weights = self.att_dic(choose_yuan_client, self.global_model_yuan, self.device)
                avg_err_weights = self.att_dic(choose_err_client, self.global_model_err, self.device)
                self.global_model_yuan.load_state_dict(avg_yuan_weights)
                self.global_model_err.load_state_dict(avg_err_weights)

                testLoss, clientMAE, MAPE, RMSE = self.evaluate(self.global_model, self.global_model_err,
                                                          self.global_model_yuan)
                self.avg_valid_client_MAE.append(clientMAE)

                if self.args.dataset == './data/metr-la':
                    threshold = 5
                elif self.args.dataset == './data/pemsd7-m':
                    threshold = 3
                if clientMAE > threshold:
                    reward = - np.exp(-clientMAE) * 10000
                else:
                    reward = np.exp(-clientMAE) * 10
                done = 0
                self.AC.store_transition(state, action.detach().cpu().numpy(), reward, state, done)
                self.AC.train()
            fig = plt.figure(figsize=(15, 12))
            plt.plot(range(1, len(self.avg_valid_client_MAE) + 1), self.avg_valid_client_MAE, label='Validation MAE')
            count = np.argmin(self.avg_valid_client_MAE) + 1
            plt.axvline(count, linestyle='--', color='r', label='Early Stopping Checkpoint')
            plt.xlabel('epochs')
            plt.ylabel('MAE')
            plt.ylim(0, 20)
            plt.xlim(0, len(self.avg_valid_client_MAE) + 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            file_name = f'Service Epoch {epoch + 1} After 100 AC MAE_plot.png'
            fig.savefig(os.path.join(self.save_folder, file_name), bbox_inches='tight')

            true_action = self.AC.select_action(state)
            self.AC.Initialize_epsilon(0.5, (epoch+1) * 10)
            true_actions = (true_action >= self.selection_threshold).float().cpu().numpy().astype(int)  # .to(torch.int)
            print(true_actions)
            with open('./true_state.txt', 'a') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"State MAE: {str(client_maes)}\n")
                f.write(f"State Loss: {str(client_losses)}\n")
                f.write(f"State TIME: {str(client_time)}\n\n")
            with open('./true_action.txt', 'a') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Action: {true_actions.tolist()}\n")
            action_np = true_action.detach().cpu().numpy()
            with open('./true_action.txt', 'a') as time_file:
                time_file.write(f"Epoch {epoch + 1}: Action {action_np}\n")
            true_choose_client = []
            true_choose_yuan_client = []
            true_choose_err_client = []
            for i in range(len(true_actions)):
                if true_actions[i] == 1:
                    true_choose_client.append(client_weights[i])
                    true_choose_yuan_client.append(client_yuan_weights[i])
                    true_choose_err_client.append(client_err_weights[i])

            len_choose = len(true_choose_client)
            with open('./true_action.txt', 'a') as time_file:
                time_file.write(f"Service Epoch {epoch + 1}: len_choose {len_choose}\n\n")
            true_avg_yuan_weights = self.att_dic(true_choose_yuan_client, self.global_model_yuan, self.device)
            true_avg_err_weights = self.att_dic(true_choose_err_client, self.global_model_err, self.device)
            self.global_model_yuan.load_state_dict(true_avg_yuan_weights)
            self.global_model_err.load_state_dict(true_avg_err_weights)

            testLoss, MAE, MAPE, RMSE = self.evaluate(self.global_model, self.global_model_err,
                                                      self.global_model_yuan)

            epoch_time_end = time.time()
            epoch_time = epoch_time_end - epoch_time_start
            with open('./FedAvg_cpu_early_time.txt', 'a') as time_file:
                time_file.write(f"Service Epoch {epoch + 1}: Total time {epoch_time}\n")
            with open('./res_MAE.txt', 'a') as time_file:
                time_file.write(
                    f"Service Epoch {epoch + 1}: TestLoss {testLoss:.6f} MAE {MAE:.6f} MAPE {MAPE:.6f} RMSE {RMSE:.6f}\n")
            print(f"Service Epoch {epoch + 1}: TestLoss {testLoss:.6f} MAE {MAE:.6f} MAPE {MAPE:.6f} RMSE {RMSE:.6f}")
            valid_loss = np.average(self.valid_losses)
            self.avg_valid_losses.append(valid_loss)
            self.avg_valid_MAE_losses.append(MAE)
            self.valid_losses = []
            early_stopping(MAE, self.global_model, self.global_model_yuan, self.global_model_err)

            if early_stopping.early_stop:
                print("Service Early stopping")
        fig = plt.figure(figsize=(15, 12))
        plt.plot(range(1, len(self.avg_valid_MAE_losses) + 1), self.avg_valid_MAE_losses, label='Validation MAE')
        count = np.argmin(self.avg_valid_MAE_losses) + 1
        plt.axvline(count, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('MAE')
        plt.ylim(0, 20)
        plt.xlim(0, len(self.avg_valid_MAE_losses) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('./pictures/Service/Service MAE_plot.png', bbox_inches='tight')

        fig = plt.figure(figsize=(15, 12))
        plt.plot(range(1, len(self.avg_valid_losses) + 1), self.avg_valid_losses, label='Validation Loss')
        minposs = np.argmin(self.avg_valid_losses) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Min Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(1, 5)
        plt.xlim(0, len(self.avg_valid_losses) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('./pictures/Service/Service loss_plot.png', bbox_inches='tight')

    def att_dic(self, client_weights, global_model, device, stepsize=1, metric=1, dp=0.001):
        global_model_params = {k: v.to(device) for k, v in global_model.state_dict().items()}

        w_next = {k: torch.zeros_like(v, device=device) for k, v in global_model_params.items()}
        att = {k: torch.zeros(len(client_weights), device=device) for k in global_model_params}

        for i, client_state_dict in enumerate(client_weights):
            for k in global_model_params.keys():
                client_tensor = client_state_dict[k].to(device)
                att[k][i] = torch.norm(global_model_params[k] - client_tensor, p=metric)

        for k in global_model_params.keys():
            att[k] = torch.nn.functional.softmax(att[k], dim=0)

        for k in global_model_params.keys():
            att_weight = torch.zeros_like(global_model_params[k], device=device)
            for i in range(len(client_weights)):
                client_state_dict = client_weights[i]
                client_tensor = client_state_dict[k].to(device)
                att_weight += (global_model_params[k] - client_tensor) * att[k][i]

            w_next[k] = global_model_params[k] - torch.mul(att_weight, stepsize)
        return w_next

    def evaluate(self, model, model_err, model_yuan):
        model = model.to(self.device)
        model_err = model_err.to(self.device)
        model_yuan = model_yuan.to(self.device)
        test_MSE, valid_losses = myutils.evaluate_model(model, model_err, model_yuan, self.err_test,
                                                        self.test, self.globel_G, self.zscore, self.err_zscore,
                                                        self.device)
        self.valid_losses.append(valid_losses)
        test_MAE, test_MAPE, test_RMSE = myutils.evaluate_metric(model, model_err, model_yuan, self.err_test,
                                                        self.test, self.globel_G, self.zscore, self.err_zscore,
                                                        self.device)
        return test_MSE, test_MAE, test_MAPE, test_RMSE


if __name__ == "__main__":
    args = get_parameters()
    mp.set_start_method('spawn')
    time_file_path = './FedAvg_cpu_early_time.txt'
    res_file_path = './res.txt'
    with open(time_file_path, 'w') as time_file:
        time_file.write('')
    with open(res_file_path, 'w') as time_file:
        time_file.write('')
    res_loss_file_path = './res_loss.txt'
    res_MAE_file_path = './res_MAE.txt'
    with open(res_loss_file_path, 'w') as time_file:
        time_file.write('')
    with open(res_MAE_file_path, 'w') as time_file:
        time_file.write('')
    DQN_action_file_path = './DQN_action.txt'
    with open(DQN_action_file_path, 'w') as time_file:
        time_file.write('')
    DQN_state_file_path = './DQN_state.txt'
    with open(DQN_state_file_path, 'w') as time_file:
        time_file.write('')
    true_action_file_path = './true_action.txt'
    with open(true_action_file_path, 'w') as time_file:
        time_file.write('')
    true_state_file_path = './true_state.txt'
    with open(true_state_file_path, 'w') as time_file:
        time_file.write('')

    res_file_path = './res_action.txt'
    with open(res_file_path, 'w') as time_file:
        time_file.write('')

    fl_service = FederatedLearningService(args)
    service_start_time = time.time()

    fl_service.train()

    service_end_time = time.time()
    time_service = service_end_time - service_start_time
    with open(time_file_path, 'a') as time_file:
        time_file.write(f"Total time: {time_service}")
    print(time_service)
