# %% [markdown]
# # Restricted Boltzmann Machine Defintion

# %%
# Import PyTorch library
import torch
import torch.nn as nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# %%
# https://github.com/khanhnamle1994/MetaRec/blob/b5e36cb579a88b32cdfb728f35f645d76b24ad95/Boltzmann-Machines-Experiments/RBM-CF-PyTorch/rbm.py#L23
# Create the Restricted Boltzmann Machine architecture
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        """
        Initialize the parameters (weights and biases) we optimize during the training process
        :param n_vis: number of visible units
        :param n_hid: number of hidden units
        """
        self.i = 0

        # Weights used for the probability of the visible units given the hidden units
        super().__init__()
        self.W = torch.zeros(n_hid, n_vis, device=device)  # torch.rand: random normal distribution mean = 0, variance = 1

        # Bias probability of the visible units is activated, given the value of the hidden units (p_v_given_h)
        self.v_bias = torch.zeros(1, n_vis, device=device)  # fake dimension for the batch = 1

        # Bias probability of the hidden units is activated, given the value of the visible units (p_h_given_v)
        self.h_bias = torch.zeros(1, n_hid, device=device)  # fake dimension for the batch = 1
    
    def lr(self):
        return 0.02

    def sample_h(self, x):
        """
        Sample the hidden units
        :param x: the dataset
        """

        # Probability h is activated given that the value v is sigmoid(Wx + a)
        # torch.mm make the product of 2 tensors
        # W.t() take the transpose because W is used for the p_v_given_h
        wx = torch.mm(x, self.W.t())
        # print(wx.shape)

        # Expand the mini-batch
        activation = wx + self.h_bias.expand_as(wx)
        # print(activation.shape)

        # Calculate the probability p_h_given_v
        p_h_given_v = torch.sigmoid(activation)

        # print("h sparse", p_h_given_v.is_sparse, torch.bernoulli(p_h_given_v).is_sparse)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_hid is activated or not activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        :param y: the dataset
        """

        # Probability v is activated given that the value h is sigmoid(Wx + a)
        wy = torch.mm(y, self.W)

        # Expand the mini-batch
        activation = wy + self.v_bias.expand_as(wy)

        # Calculate the probability p_v_given_h
        p_v_given_h = torch.sigmoid(activation)

        # print("v sparse", p_v_given_h.is_sparse, torch.bernoulli(p_v_given_h).is_sparse)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_vis is activated or not activated
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train_model(self, v0, vk, ph0, phk):
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        w_extra = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        v_extra = torch.sum((v0 - vk), 0)
        h_extra = torch.sum((ph0 - phk), 0)

        # if self.i % 45 == 0:
            # print(torch.max(w_extra), torch.max(v_extra), torch.max(h_extra), flush=True)

        # Approximate the gradients with the CD algorithm
        # TODO learning rate toevoegen
        self.W += self.lr() * w_extra

        # Add (difference, 0) for the tensor of 2 dimensions
        self.v_bias += self.lr() * v_extra
        self.h_bias += self.lr() * h_extra
        self.i += 1

# %%
cuda = torch.device('cuda')

# %% [markdown]
# # General Imports

# %%
import numpy as np
import pandas as pd
import scipy
import sklearn
import gzip
import json
from tqdm import tqdm
import os
from collections import Counter
from datetime import datetime
import math
tqdm.pandas() #for progres_apply etc.

# %%
#read file line-by-line and parse json, returns dataframe
def parse_json(filename_gzipped_python_json, read_max=-1):
  #read gzipped content
  f=gzip.open(filename_gzipped_python_json,'r')
  
  #parse json
  parse_data = []
  for line in tqdm(f): #tqdm is for showing progress bar, always good when processing large amounts of data
    line = line.decode('utf-8')
    line = line.replace('true','True') #difference json/python
    line = line.replace('false','False')
    parsed_result = eval(line) #load python nested datastructure
    # print(filename_gzipped_python_json == steam_path + steam_reviews and 'user_id' not in parsed_result)
    # break
    if filename_gzipped_python_json == steam_path + steam_reviews and 'user_id' not in parsed_result:
      continue
      
    parse_data.append(parsed_result)
    if read_max !=-1 and len(parse_data) > read_max:
      print(f'Break reading after {read_max} records')
      break
  print(f"Reading {len(parse_data)} rows.")

  #create dataframe
  df= pd.DataFrame.from_dict(parse_data)
  return df

# %%
steam_path = './data/'
metadata_games = 'steam_games.json.gz' 
user_items = 'australian_users_items.json.gz'
user_reviews = 'australian_user_reviews.json.gz'
game_bundles = 'bundle_data.json.gz'
steam_reviews= 'steam_reviews.json.gz'

# %%
torch.manual_seed(0)

# %% [markdown]
# Train model

# %%
def score_model(rbm, batch_size, train_matrix, test_matrix):
    test_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
    s = 0  # a counter (float type) 
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0] - batch_size, batch_size):
        v = train_matrix[id_user:id_user + batch_size]  # training set inputs are used to activate neurons of my RBM
        vt = test_matrix[id_user:id_user + batch_size]  # target
        # v = convert_sparse_matrix_to_sparse_tensor(training_sample)
        # vt = convert_sparse_matrix_to_sparse_tensor(training_sample2)
        v = v.todense()
        vt = vt.todense()

        # v = v.to_dense()
        # vt = vt.to_dense()
        v = v - 1
        vt = vt - 1
        v = torch.Tensor(v)
        vt = torch.Tensor(vt)
        if torch.cuda.is_available():
            v = v.cuda()
            vt = vt.cuda()
        if len(vt[vt > -1]) > 0:
            _, h = rbm.sample_h(v)
            v, _ = rbm.sample_v(h)

            # Update test RMSE reconstruction error
            test_recon_error += torch.sqrt(torch.mean((vt[vt > -1] - v[vt > -1])**2)) * len(vt > -1)
            s += len(vt > -1) 

    return test_recon_error / s


# https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
def convert_sparse_matrix_to_sparse_tensor(X):
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

def create_rbm(train_matrix, test_matrix, n_hidden, batch_size, epochs, rbm=None) -> RBM:
    n_vis = train_matrix.shape[1]
    train_errors = []
    test_errors = []
    if rbm is None:
        rbm = RBM(n_vis, n_hidden)

    print("start training")
    for epoch in tqdm(range(epochs)):
        rbm.train()
        train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
        s = 0
        
        for user_id in range(0, train_matrix.shape[0] - batch_size, batch_size):
            training_sample = train_matrix[user_id : user_id + batch_size]
            training_sample2 = train_matrix[user_id : user_id + batch_size]
            # print(training_sample)
            v0 = convert_sparse_matrix_to_sparse_tensor(training_sample)
            # print(v0.coalesce().indices())
            vk = convert_sparse_matrix_to_sparse_tensor(training_sample2)

            v0 = v0.to_dense()
            vk = vk.to_dense()
            v0 = v0.sub(1)
            vk = vk.sub(1)
            
            ph0, _ = rbm.sample_h(v0)

            # Third for loop - perform contrastive divergence
            # TODO misschien is iets lager proberen?
            for k in range(1):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)

                # We don't want to learn when there is no rating by the user, and there is no update when rating = -1
                # Remove indices from vk vector that are not in the v0 vector => get sparse tensor again
                vk[v0 < 0] = v0[v0 < 0]
                

            phk, _ = rbm.sample_h(vk)

            rbm.train_model(v0, vk, ph0, phk)
            vk, _ = rbm.sample_v(hk)
            
            train_recon_error += torch.sqrt(torch.mean((v0[v0 > -1] - vk[v0 > -1])**2)) * len(v0 > -1)
            s += len(v0 > -1)
            
            # print((torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t().shape)
            # print(torch.sum((-vk + v0), 0).shape)
            # print(torch.sum((ph0 - phk), 0).shape)
            
        train_errors.append(train_recon_error / s)

        # print('calculating test scores')
        rbm.eval()
        test_errors.append(score_model(rbm, batch_size, train_matrix, test_matrix))

        # print('finished epoch', epoch)    

    import matplotlib.pyplot as plt
    # Plot the RMSE reconstruction error with respect to increasing number of epochs
    plt.plot(torch.Tensor(train_errors, device='cpu'), label="train")
    plt.plot(torch.Tensor(test_errors, device='cpu'), label="test")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'bigboi-{n_hidden}-{batch_size}-{epochs}.jpg')

    return rbm

# Evaluate the RBM on test set
# test_recon_error = score_model(rbm)
# print("Final error", test_recon_error)


# %% [markdown]
# Hitrate / Recall

# %%
def compute_hr(train_matrix, test_matrix, rbm, k=10, batch_size=100):
    s = 0  # a counter (float type) 
    hitrates = []
    recall = []
    nDCG = []
    # for loop - go through every single user
    for id_user in range(0, train_matrix.shape[0]): # - batch_size, batch_size):
        v = train_matrix[id_user]  # training set inputs are used to activate neurons of my RBM
        vt = test_matrix[id_user]  # target

        target_data = vt.data
        target_index = vt.indices
        target_recommendations = target_index[target_data == 2]
        # print(target_test)

        v = v.todense()

        v = v - 1
        v = torch.Tensor(v)
        if torch.cuda.is_available():
            v = v.cuda()
        
        if len(target_recommendations) > 0: # check that target contains recommendations (only needed for aussies)
            _, h = rbm.sample_h(v)
            recommended, _ = rbm.sample_v(h)

            # all recommendations
            _, indices =  torch.topk(recommended[v < 0], k)
            recommendations = torch.tensor(indices, device='cpu').tolist()

            counter = 0
            total = len(target_recommendations)
            for target in target_recommendations:
                if target in recommendations:
                    counter += 1
            # counter = len(recommendations)

            recall.append(counter / total)
            hitrates.append(min(1, counter))

            # nDCG
            idcg = np.sum([1 / np.log2(i+2) for i in range(min(k, len(target_recommendations)))])
            dcg = 0
            for i, r in enumerate(recommendations):
                if r in target_recommendations:
                    dcg += 1 / np.log2(i+2)

            nDCG.append(dcg / idcg) 

    return hitrates, recall, nDCG

# %%
def recommend(rbm, v, vt, k, p=True):
    target_data = vt.data
    target_index = vt.indices
    target_recommendations = target_index[target_data == 2]
    v = v.todense()
    v = v - 1
    v = torch.Tensor(v)
    if torch.cuda.is_available():
        v = v.cuda()
    
    _, h = rbm.sample_h(v)
    recommended, _ = rbm.sample_v(h)

    # all recommendations
    values, indices =  torch.topk(recommended[v < 0], k)
    recommendations = torch.tensor(indices, device='cpu').tolist()

    if p:
        print('20', recommended[0][20])
        print('21', recommended[0][21])
        print("average value", torch.mean(recommended[0]))

    found = True
    for r in recommendations:
        if r in target_recommendations:
            if p:
                print("HIT")
            found = True
            break

    if found and p:
        print("values", values)
        print("recommended", recommendations)
        print("real", target_recommendations)

    
    
    return recommendations


# %% [markdown]
# # Reading Full Dataset

# %%
steam_reviews_df = parse_json(steam_path + steam_reviews)
steam_reviews_df_small = steam_reviews_df[['user_id', 'product_id', 'recommended', 'date']]

# %%
steam_reviews_df_cleaned = steam_reviews_df_small.dropna(axis=0, subset=['user_id'])

# %%
steam_reviews_df_cleaned.head(5)
steam_reviews_df["user_id"].value_counts(dropna=False)

# %%
dct = {}
def map_to_consecutive_id(uuid):
  if uuid in dct:
    return dct[uuid]
  else:
    id = len(dct)
    dct[uuid] = id
    return id
steam_reviews_df_cleaned['product_id_int'] = steam_reviews_df_cleaned['product_id'].progress_apply(map_to_consecutive_id)

# %% [markdown]
# ## Date Split

# %%
steam_reviews_df_cleaned["date"] = pd.to_datetime(steam_reviews_df_cleaned["date"])


# %%
steam_reviews_df_grouped = steam_reviews_df_cleaned.groupby("user_id")[["product_id_int", "recommended", "date"]].agg(list)
steam_reviews_df_grouped_smaller = steam_reviews_df_grouped[steam_reviews_df_grouped["recommended"].map(len) > 1]

# %%
dct.clear()
steam_reviews_df_grouped_smaller = steam_reviews_df_grouped_smaller.reset_index()
steam_reviews_df_grouped_smaller["user_id_int"] = steam_reviews_df_grouped_smaller["user_id"].progress_apply(map_to_consecutive_id)

# %%
print(steam_reviews_df_grouped.shape[0] - steam_reviews_df_grouped_smaller.shape[0])

# %%

def split(items, train_percentage):
    train_count = math.floor(len(items) * train_percentage)
    return items[0:train_count], items[train_count:]

train_percentage = 0.8
steam_reviews_df_grouped_smaller["product_history"] = steam_reviews_df_grouped_smaller["product_id_int"].progress_apply(lambda items: split(items, train_percentage)[0])
steam_reviews_df_grouped_smaller["product_future"] = steam_reviews_df_grouped_smaller["product_id_int"].progress_apply(lambda items: split(items, train_percentage)[1])
steam_reviews_df_grouped_smaller["recommended_history"] = steam_reviews_df_grouped_smaller["recommended"].progress_apply(lambda items: split(items, train_percentage)[0])
steam_reviews_df_grouped_smaller["recommended_future"] = steam_reviews_df_grouped_smaller["recommended"].progress_apply(lambda items: split(items, train_percentage)[1])

# %%
steam_reviews_df_grouped_smaller["recommended"].map(len).describe()
steam_reviews_df_grouped_smaller["recommended"].map(len).sum()

# %%
#Create scipy csr matrix
def get_sparse_matrix(df, shape, recommended_col="recommended_history", product_col="product_history"):
    user_ids = []
    product_ids = []
    values = []
    for _, row in df.iterrows():
        products = row[product_col]
        user = row['user_id_int']
    
        recommended = row[recommended_col]
        user_ids.extend([user] * len(products))
        product_ids.extend(products)
        values.extend([2 if recommended[i] else 1 for i in range(len(products))])
    #create csr matrix
    # values = np.ones(len(user_ids))
    matrix = scipy.sparse.csr_matrix((values, (user_ids, product_ids)), shape=shape, dtype=np.int32)
    return matrix

# %%
steam_reviews_set = steam_reviews_df_grouped_smaller#.head(100000)

# %%
shape = (steam_reviews_set.shape[0], steam_reviews_df_cleaned['product_id_int'].max() + 1)

steam_reviews_set = steam_reviews_set.reset_index()
train_matrix_big = get_sparse_matrix(steam_reviews_set, shape)
test_matrix_big = get_sparse_matrix(steam_reviews_set, shape, recommended_col="recommended_future", product_col="product_future")


# %% [markdown]
# # Model RBM

# %%
rbm_big_10 = create_rbm(train_matrix_big, test_matrix_big, 1024, 10240, 10)

# %%
print("10 epochs")
hr, r, ndcg = compute_hr(train_matrix_big, test_matrix_big, rbm_big_10)
print("hr", np.average(hr))
print("recall", np.average(r))
print("ndcg", np.average(ndcg))