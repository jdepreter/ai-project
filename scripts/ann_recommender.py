import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, device

import numpy as np
from scipy import sparse
import pandas as pd
from tqdm import trange


class Network(nn.Module):
    def __init__(self, input_size, output_size, dropout, l1, l2, l3, activation=F.relu):
        super().__init__()
        
        # use 3 layers and fc layer
        self.activation = activation

        self.dropout = nn.Dropout(dropout)
        if torch.cuda.is_available():
            self.device = device("cuda")

        self.lin1 = nn.Linear(input_size, l1).to(self.device)
        self.lin2 = nn.Linear(l1, l2).to(self.device)
        self.lin3 = nn.Linear(l2, l3).to(self.device)
        self.fc = nn.Linear(l3, output_size).to(self.device)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.to(self.device)

        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))

        x = self.dropout(x)
        output = self.fc(x)
        return output


def params_to_str(params):
    return "-".join([s.__name__ if callable(s) else str(s) for s in params] )

def score_model(rbm, batch_size, train_df, train_matrix, test_matrix):
    test_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
    s = 0  # a counter (float type) 
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0] - batch_size, batch_size):
        v = train_matrix[id_user:id_user + batch_size]  # training set inputs are used to activate neurons of my RBM
        v = convert_to_user_tensor(v, train_df[id_user:id_user + batch_size])

        vt = test_matrix[id_user:id_user + batch_size]  # target
        vt = convert_sparse_matrix_to_sparse_tensor(vt)
        vt = vt.to_dense()
        vt = vt.sub(1)
        if torch.cuda.is_available():
            vt = vt.cuda()

        if len(vt[vt > -1]) > 0:
            vk = rbm(v)
            # Update test RMSE reconstruction error
            loss = torch.sqrt(torch.mean((vt[vt > -1] - vk[vt > -1])**2))
            loss.backward()
            test_recon_error += loss
            s += 1

    return test_recon_error / s 


# https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
def convert_sparse_matrix_to_sparse_tensor(X, k=5):
    coo = X.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    # print(values)
    # print("values", v)
    shape = coo.shape
    tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)) 
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor 

def convert_to_user_tensor(X: sparse.csr_matrix, users: pd.DataFrame):
    coo = X.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)) 
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    tensor = tensor.to_dense()
    tensor = tensor.sub(1)

    if users is not None:
        users_tensor = torch.Tensor(users.values)
        if torch.cuda.is_available():
            users_tensor = users_tensor.cuda()
        tensor = torch.cat([tensor, users_tensor], dim=1)
    return tensor 

    
def create_model(train_df: pd.DataFrame, train_matrix, test_matrix, batch_size, epochs, params, model=None, k=5, hrmod=20, lr=0.02, stop=False):
    n_vis = train_matrix.shape[1] + train_df.shape[1]
    train_errors = []
    test_errors = []
    if model is None:
        model = Network(n_vis, train_matrix.shape[1], *params)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    metrics = {
        "hr": [],
        "r": [],
        "ndcg": []
    }

    # print("start training")
    for epoch in trange(epochs):
        model.train()
        train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
        s = 0
        
        for user_id in range(0, train_matrix.shape[0] - batch_size, batch_size):
            training_sample = train_matrix[user_id : user_id + batch_size]
            v0 = convert_to_user_tensor(training_sample, train_df[user_id: user_id + batch_size])
            
            optim.zero_grad()            
            vk = model(v0)
            ratings = v0[:,:-train_df.shape[1]]
            loss = torch.sqrt(torch.mean((ratings[ratings > -1] - vk[ratings > -1])**2))
            loss.backward()
            optim.step()
            train_recon_error +=loss
            s += 1
            
        train_errors.append(train_recon_error / s)

        model.eval()
        test_errors.append(score_model(model, batch_size, train_df, train_matrix, test_matrix))

        if epoch % hrmod == hrmod - 1:
            hr, r, ndcg = compute_hr(train_df, train_matrix, test_matrix, model, batch_size=batch_size)
            metrics["hr"].append(np.average(hr))
            metrics["r"].append(np.average(r))
            metrics["ndcg"].append(np.average(ndcg))
            if stop and len(metrics["hr"]) > 2:
                hr1 = metrics["hr"][-3]
                hr2 = metrics["hr"][-2]
                hrnow = metrics["hr"][-1]
                if hrnow < hr1 and hrnow < hr2:
                    print("Hr decreasing => stopping training")
                    break


    import matplotlib.pyplot as plt
    # Plot the RMSE reconstruction error with respect to increasing number of epochs
    plt.plot(np.arange(1, len(train_errors), 1), torch.Tensor(train_errors[1:], device='cpu'), label="train")
    plt.plot(np.arange(1, len(train_errors), 1), torch.Tensor(test_errors[1:], device='cpu'), label="test")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()


    plt.savefig(f'ann-{params_to_str(params)}-{batch_size}-{epochs}.jpg')
    plt.show()
    plt.clf()

    return model, metrics


def compute_hr(train_df, train_matrix, test_matrix, rbm, k=10, batch_size=100):
    hitrates = []
    recall = []
    nDCG = []
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0] - batch_size, batch_size): # - batch_size, batch_size):
        vt = test_matrix[id_user:id_user + batch_size]  # target
        if vt.getnnz() == 0:
            continue

        vt = convert_sparse_matrix_to_sparse_tensor(vt)
        vt = vt.to_dense()
        vt = vt.sub(1)
        if torch.cuda.is_available():
            vt = vt.cuda()

        # ground truth
        users, movies = (vt > 1).nonzero(as_tuple=True)

        indices = torch.stack([users, movies])
        shape = (batch_size, train_matrix.shape[1])
        target = torch.sparse.LongTensor(indices, torch.add(vt[vt > 1].flatten(), 1), torch.Size(shape))
        target_dense = target.to_dense()

        target_rating, target_movie = torch.topk(target_dense, k, 1)
        # target_movie[target_rating < 3] = -1 # remove all bad movies from top k

        values, _ = torch.max(target_rating, dim=1)
        users_with_target = (values > 0).nonzero(as_tuple=True)[0].cpu().tolist()


        # predicted
        v = train_matrix[id_user:id_user + batch_size]  # training set inputs are used to activate neurons of my RBM
        v = convert_to_user_tensor(v, train_df[id_user:id_user + batch_size])
        ratings = v[:,:-3]
        recommended = rbm(v)
        recommended[ratings != -1] = -10
        predicted_rating, predicted_movie = torch.topk(recommended, k)

        # TODO optimize range s.t. users without target are skipped
        for user in users_with_target:

            # all recommendations
            user_target = target_movie[user][target_rating[user] > 0].cpu().tolist()
            user_pred = predicted_movie[user].cpu().tolist()

            counter = 0
            total = min(k, len(user_target))
            for target in user_target:
                if target in user_pred:
                    counter += 1
            # counter = len(recommendations)

            recall.append(counter / total)
            hitrates.append(min(1, counter))

            # nDCG
            idcg = np.sum([1 / np.log2(i+2) for i in range(min(k, len(user_target)))])
            dcg = 0
            for i, r in enumerate(user_pred):
                if r in user_target:
                    dcg += 1 / np.log2(i+2)

            nDCG.append(dcg / idcg) 

    return hitrates, recall, nDCG