#loaddata.py
import math
import numpy as np
import pandas as pd
import torch
import torch.utils as utils
import os
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def split_dataset(csv_file_path, len_train, len_val):
    """
        Parameters:
            csv_file_path (str): Path to the CSV file.
            len_train (int): Size of the training set (number of rows).
            len_val (int): Size of the validation set (number of rows).
        Returns:
            tuple: A tuple containing three elements, which are the DataFrame objects for the training set, validation set, and test set.
    """
    vel = pd.read_csv(csv_file_path, header=None)
    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    """
        Parameters:
            data (pd.DataFrame): Raw time series data.
            n_his (int): Number of historical data points used as model input features.
            n_pred (int): Number of future data points to predict, serving as the model's output target.
            device (str): Device to which the data should be loaded, e.g., 'cpu' or 'cuda'.
        Returns:
            tuple: PyTorch tensors containing the transformed input features and target outputs.
    """
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    if isinstance(data, np.ndarray):
        data_np = data
    else:
        data_np = data.to_numpy()
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data_np[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data_np[tail + n_pred - 1]
    return torch.Tensor(x), torch.Tensor(y)

def split_columns_into_groups(data, num):
    cols_per_group = data.shape[1] // num
    split_data = []
    client_n_vertices = []
    for i in range(num - 1):
        start_col = i * cols_per_group
        end_col = (i + 1) * cols_per_group
        split_data.append(data[:, start_col:end_col])

    start_col = (num - 1) * cols_per_group
    end_col = data.shape[1]
    split_data.append(data[:, start_col:end_col])

    for i, group_data in enumerate(split_data):
        print(f"Group {i + 1} Data Size:", group_data.shape)
        client_n_vertices.append(group_data.shape[1])
    print(client_n_vertices)
    return split_data, client_n_vertices

def Greph(vel):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(vel)
    pca = PCA()
    df_pca = pca.fit_transform(df_std)
    df_principal = pd.DataFrame(data=df_pca.astype(np.float32))
    num_samples, num_components = df_principal.shape
    c_vectors = df_principal.values.T
    norms = np.linalg.norm(c_vectors, axis=1)
    normalized_c_vectors = c_vectors / norms[:, np.newaxis]
    cosine_similarities = np.dot(normalized_c_vectors, normalized_c_vectors.T)
    cosine_similarities = np.maximum(cosine_similarities, 0)

    return cosine_similarities

def data_preparate(csv_file_path, n_his, n_pred, num, batch_size, device):
    filename = 'vel.csv'
    file_path = os.path.join(csv_file_path, filename)
    data_col = pd.read_csv(file_path, header=None).shape[0]
    print(data_col)

    # Define the sizes of the validation and test sets.
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    print(len_train)
    print(len_test)
    print(len_val)

    train = []
    val = []
    test = []
    zscore_client = []

    train_temp, val_temp, test_temp = split_dataset(file_path, len_train, len_val)
    err_train_temp = np.diff(train_temp, axis=0)
    err_val_temp = np.diff(val_temp, axis=0)
    err_test_temp = np.diff(test_temp, axis=0)

    train_temp_all = train_temp
    val_temp_all = val_temp
    test_temp_all = test_temp

    zscore = preprocessing.StandardScaler()
    train_temp_all = zscore.fit_transform(train_temp_all)
    val_temp_all = zscore.transform(val_temp_all)
    test_temp_all = zscore.transform(test_temp_all)

    train_temp,_ = split_columns_into_groups(train_temp.to_numpy(), num)
    val_temp,_ = split_columns_into_groups(val_temp.to_numpy(), num)
    test_temp,_ = split_columns_into_groups(test_temp.to_numpy(), num)
    for i in range(num):
        zscore_temp = preprocessing.StandardScaler()
        train_temp[i] = zscore_temp.fit_transform(train_temp[i])
        val_temp[i] = zscore_temp.transform(val_temp[i])
        test_temp[i] = zscore_temp.transform(test_temp[i])
        zscore_client.append(zscore_temp)

        x_train, y_train = data_transform(train_temp[i], n_his, n_pred, device)
        train_data = utils.data.TensorDataset(x_train, y_train)
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
        train.append(train_iter)

        x_val, y_val = data_transform(val_temp[i], n_his, n_pred, device)
        val_data = utils.data.TensorDataset(x_val, y_val)
        val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        val.append(val_iter)

        x_test, y_test = data_transform(test_temp[i], n_his, n_pred, device)
        test_data = utils.data.TensorDataset(x_test, y_test)
        test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        test.append(test_iter)

    x_test, y_test = data_transform(test_temp_all, n_his, n_pred, device)
    test_data = utils.data.TensorDataset(x_test, y_test)
    yuan_test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    err_train = []
    err_val = []
    err_test = []
    err_zscore_client = []
    err_train_temp_all = err_train_temp
    err_val_temp_all = err_val_temp
    err_test_temp_all = err_test_temp
    err_zscore = preprocessing.StandardScaler()
    err_train_temp_all = err_zscore.fit_transform(err_train_temp_all)
    err_val_temp_all = err_zscore.transform(err_val_temp_all)
    err_test_temp_all = err_zscore.transform(err_test_temp_all)

    err_train_temp, _ = split_columns_into_groups(err_train_temp, num)
    err_val_temp, _ = split_columns_into_groups(err_val_temp, num)
    err_test_temp, client_n_vertices = split_columns_into_groups(err_test_temp, num)
    for i in range(num):
        zscore_temp = preprocessing.StandardScaler()
        err_train_temp[i] = zscore_temp.fit_transform(err_train_temp[i])
        err_val_temp[i] = zscore_temp.transform(err_val_temp[i])
        err_test_temp[i] = zscore_temp.transform(err_test_temp[i])
        err_zscore_client.append(zscore_temp)

        x_train, y_train = data_transform(err_train_temp[i], n_his-1, n_pred, device)
        train_data = utils.data.TensorDataset(x_train, y_train)
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
        err_train.append(train_iter)

        x_val, y_val = data_transform(err_val_temp[i], n_his-1, n_pred, device)
        val_data = utils.data.TensorDataset(x_val, y_val)
        val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        err_val.append(val_iter)

        x_test, y_test = data_transform(err_test_temp[i], n_his-1, n_pred, device)
        test_data = utils.data.TensorDataset(x_test, y_test)
        test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        err_test.append(test_iter)

    x_test, y_test = data_transform(err_test_temp_all, n_his-1, n_pred, device)
    test_data = utils.data.TensorDataset(x_test, y_test)
    err_test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train, val, test, yuan_test_iter, zscore, zscore_client, err_train, err_val, err_test, err_test_iter, err_zscore, err_zscore_client, client_n_vertices
def load_sparse_adjacency_matrix(csv_file, num):
    save_dir = './temp'
    os.makedirs(save_dir, exist_ok=True)

    filename = 'vel.csv'
    file_path = os.path.join(csv_file, filename)
    graph_vel = pd.read_csv(file_path, header=None)
    globel_G = Greph(graph_vel)

    data_col = graph_vel.shape[1]
    len_clo = data_col // num
    client_file = []
    for i in range(num):
        start_col = len_clo * i
        if i == num - 1:
            end_col = data_col
        else:
            end_col = len_clo * (i + 1)
        client_file.append(graph_vel.iloc[:, start_col:end_col].copy())

    if csv_file == './data/metr-la':
        np.save('./temp/globel_graph_metr.npy', globel_G)
        globel_G = np.load('./temp/globel_graph_metr.npy')
        client_graph = []
        for i in range(num):
            temp = Greph(client_file[i])
            client_graph.append(temp)
            filename = f'./temp/metr_la_client{i}.npz'
            np.save(filename, temp)
        n_vertex = 207
    elif csv_file == './data/pemsd7-m':
        np.save('./temp/globel_graph_pemsd7-m.npy', globel_G)
        globel_G = np.load('./temp/globel_graph_pemsd7-m.npy')
        client_graph = []
        for i in range(num):
            temp = Greph(client_file[i])
            client_graph.append(temp)
            filename = f'./temp/pems_client{i}.npz'
            np.save(filename, temp)
        n_vertex = 228

    return globel_G, client_graph, n_vertex

