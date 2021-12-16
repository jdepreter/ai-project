# %% [markdown]
# # Restricted Boltzmann Machine Defintion

# %%
# Import PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, flatten, device
from tqdm import tqdm, trange
from scipy import sparse
import json

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
# https://github.com/khanhnamle1994/MetaRec/blob/b5e36cb579a88b32cdfb728f35f645d76b24ad95/Boltzmann-Machines-Experiments/RBM-CF-PyTorch/rbm.py#L23
# Create the Restricted Boltzmann Machine architecture
class network(nn.Module):
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

# %%
cuda = torch.device('cuda')

# %% [markdown]
# # General Imports

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas() #for progres_apply etc.

# %% [markdown]
# # Reading

# %%
def load_interactions(path, n_splits=5):
    df = pd.read_pickle(os.path.join(os.getcwd(), path))
    df[['interactions', 'train', 'val', 'test']] = df[['interactions', 'train', 'val', 'test']].applymap(lambda x: np.array(x, dtype=np.int32))
    interactions_dict = {}
    for split in trange(n_splits):
        for column in ['train', 'val', 'test']:
            interactions_dict[split, column] = pd.DataFrame({
                'user_id': df['user_id'],
                'steam_id': df['steam_id'],
                'item_id': df[column].apply(lambda x: x[split, 0]),
                'playtime_forever': df[column].apply(lambda x: x[split, 1]),
                'playtime_2weeks': df[column].apply(lambda x: x[split, 2])})
    return interactions_dict

# %%
interactions = load_interactions("./data-cleaned/interactions_splits.pkl.gz")
interactions[0, 'train'].head()
games = pd.read_pickle(os.path.join(os.getcwd(), "./data-cleaned/games.pkl.gz"))
games.head()

# %%
train0 = interactions[0, 'train']
test0 = interactions[0, 'test']

# %%
train0["item_id"].map(len).describe()

# %%
train0.iloc[100,:]

# %% [markdown]
# ## Sparse Matrix

# %%
def to_simple_rating(playtime):
    if playtime < 120:
        # less than 2 hrs
        return 1
    elif playtime < 240:
        # less than 4 hrs
        return 2
    elif playtime < 600:
        # less than 10 hrs
        return 3
    elif playtime < 24*60:
        # less than 24 hrs
        return 4
    else:
        return 5


def log_playtime(playtime):
    return np.log2(playtime)

# %%

#Create scipy csr matrix
def get_sparse_matrix(df):
    shape = (df.shape[0], games.shape[0])
    
    user_ids = []
    item_ids = []
    values = []
    for idx, row in df.iterrows():
        items = row['item_id']
        user = idx
        score = row["playtime_forever"] + 2* row["playtime_2weeks"]
        
        # recommended = row['recommended']
        user_ids.extend([user] * len(items))
        item_ids.extend(items)
        values.extend([to_simple_rating(score[i]) for i in range(len(items))])
    # create csr matrix
    # values = np.ones(len(user_ids))
    matrix = sparse.csr_matrix((values, (user_ids, item_ids)), shape=shape, dtype=np.int32)
    return matrix

# %%
# test_matrix = get_sparse_matrix(test0)
# train_matrix = get_sparse_matrix(train0)
# train_matrix


# %% [markdown]
# # Evaluation

# %%
def compute_hr(train_matrix, test_matrix, rbm, k=10, batch_size=100):
    hitrates = []
    recall = []
    nDCG = []
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0] - batch_size, batch_size): # - batch_size, batch_size):
        v = train_matrix[id_user:id_user + batch_size]  # training set inputs are used to activate neurons of my RBM
        vt = test_matrix[id_user:id_user + batch_size]  # target
        if vt.getnnz() == 0:
            continue

        v = convert_sparse_matrix_to_sparse_tensor(v)
        vt = convert_sparse_matrix_to_sparse_tensor(vt)
        v = v.to_dense()
        vt = vt.to_dense()
        v = v.sub(1)
        vt = vt.sub(1)

        if torch.cuda.is_available():
            vt = vt.cuda()
            v = v.cuda()

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
        # _, h = rbm.sample_h(v)
        # recommended, _ = rbm.sample_v(h)
        recommended = rbm(v)
        recommended[v != -1] = -10
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

# %% [markdown]
# # Train model

# %%
def params_to_str(params):
    return "-".join([s.__name__ if callable(s) else str(s) for s in params] )

def score_model(rbm, batch_size, train_matrix, test_matrix):
    test_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
    s = 0  # a counter (float type) 
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0] - batch_size, batch_size):
        v = train_matrix[id_user:id_user + batch_size]  # training set inputs are used to activate neurons of my RBM
        vt = test_matrix[id_user:id_user + batch_size]  # target
        v = convert_sparse_matrix_to_sparse_tensor(v)
        vt = convert_sparse_matrix_to_sparse_tensor(vt)

        v = v.to_dense()
        vt = vt.to_dense()
        v = v.sub(1)
        vt = vt.sub(1)
        
        if torch.cuda.is_available():
            v = v.cuda()
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
    
def create_rbm(train_matrix, test_matrix, batch_size, epochs, params, model=None, k=5, hrmod=20):
    n_vis = train_matrix.shape[1]
    train_errors = []
    test_errors = []
    if model is None:
        model = network(n_vis, n_vis, *params)
    optim = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    metrics = {
        "hr": [],
        "r": [],
        "ndcg": []
    }

    print("start training")
    for epoch in trange(epochs):
        model.train()
        train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
        s = 0
        
        for user_id in range(0, train_matrix.shape[0] - batch_size, batch_size):
            training_sample = train_matrix[user_id : user_id + batch_size]
            v0 = convert_sparse_matrix_to_sparse_tensor(training_sample)

            v0 = v0.to_dense()
            v0 = v0.sub(1)
            
            optim.zero_grad()            
            vk = model(v0)
            loss = torch.sqrt(torch.mean((v0[v0 > -1] - vk[v0 > -1])**2))
            loss.backward()
            optim.step()
            train_recon_error +=loss
            s += 1
            
        train_errors.append(train_recon_error / s)

        model.eval()
        test_errors.append(score_model(model, batch_size, train_matrix, test_matrix))

        if epoch % hrmod == hrmod - 1:
            hr, r, ndcg = compute_hr(train_matrix, test_matrix, model, batch_size=batch_size)
            metrics["hr"].append(np.average(hr))
            metrics["r"].append(np.average(r))
            metrics["ndcg"].append(np.average(ndcg))


    import matplotlib.pyplot as plt
    # Plot the RMSE reconstruction error with respect to increasing number of epochs
    plt.plot(torch.Tensor(train_errors, device='cpu'), label="train")
    plt.plot(torch.Tensor(test_errors, device='cpu'), label="test")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()


    plt.savefig(f'ann-{params_to_str(params)}-{batch_size}-{epochs}.jpg')
    plt.show()
    plt.clf()

    return model, metrics

# Evaluate the RBM on test set
# test_recon_error = score_model(rbm)
# print("Final error", test_recon_error)


# %% [markdown]
# rbm

# %%
# rbm10 = create_rbm(train_matrix, test_matrix, 1000, 10000, 200)

# %%
# torch.save(rbm10.state_dict(), "./ann-20-50-20-steam400-train0")

# %% [markdown]
# hr

# %%
# hr, r, ndcg = compute_hr(train_matrix, test_matrix, rbm10, k=10, batch_size=15000)

# %%
# print(np.average(hr), np.average(r), np.average(ndcg))

# %%
# rbm = RBM(n_vis, n_hidden)
# rbm.load_state_dict(torch.load("./network"))
# rbm.eval()

# %% [markdown]
# # Hyperparam Tuning

# %%
train_matrix = get_sparse_matrix(train0)
test_matrix = get_sparse_matrix(test0)

# %%
l1s = np.arange(16, 33, 16)
l2s = np.arange(24, 65, 8)
l3s = np.arange(24, 129, 16)
# l1s = [48]
# l2s = [32, 48]
# dropouts = [0.1, 0.2, 0.3]
dropouts = [0.1]

epochs = 100
activation = torch.tanh

for l1 in l1s:
    for l2 in l2s:
        for l3 in l3s:
            for dropout in dropouts:
                model, metrics = create_rbm(train_matrix, test_matrix, 10000, epochs, (dropout, l1, l2, l3, activation), hrmod=20)
                
                torch.save(model.state_dict(), f"./ann-rating-{l1}-{l2}-{l3}-{dropout}-steam{epochs}-train0")
                with open(f"metrics-rating-{l1}-{l2}-{l3}-{dropout}.json", "w") as f:
                    f.write(json.dumps(metrics))


