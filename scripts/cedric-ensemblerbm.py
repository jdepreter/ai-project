# %% [markdown]
# # Restricted Boltzmann Machine Defintion

# %% [markdown]
# ### Import PyTorch library

# %%
import torch
import torch.nn as nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# %% [markdown]
# ### the Restricted Boltzmann Machine architecture

# %%
# https://github.com/khanhnamle1994/MetaRec/blob/b5e36cb579a88b32cdfb728f35f645d76b24ad95/Boltzmann-Machines-Experiments/RBM-CF-PyTorch/rbm.py#L23
# 
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k, batch_size):
        """
        Initialize the parameters (weights and biases) we optimize during the training process
        :param n_vis: number of visible units
        :param n_hid: number of hidden units
        """
        self.i = 0
        self.K = k
        self.batch_size = batch_size

        # Weights used for the probability of the visible units given the hidden units
        super().__init__()
        self.W = torch.randn(k, n_hid, n_vis)  # torch.rand: random normal distribution mean = 0, variance = 1

        # Bias probability of the visible units is activated, given the value of the hidden units (p_v_given_h)
        self.v_bias = torch.zeros(k, 1, n_vis)  # fake dimension for the batch = 1

        # Bias probability of the hidden units is activated, given the value of the visible units (p_h_given_v)
        self.h_bias = torch.zeros(1, n_hid)  # fake dimension for the batch = 1

        if torch.cuda.is_available():
            self.W = self.W.cuda()
            self.v_bias = self.v_bias.cuda()
            self.h_bias = self.h_bias.cuda()
    
    def lr(self):
        """
        return the learning rate of the model, lr is based on batchsize
        :return: constant/batch_size
        """
        return 0.01 / self.batch_size

    def sample_h(self, x):
        """
        Sample the hidden units
        :param x: the dataset
        """

        # Probability h is activated given that the value v is sigmoid(Wx + a)
        # torch.mm make the product of 2 tensors
        # W.t() take the transpose because W is used for the p_v_given_h

        temp = torch.transpose(self.W, 1, 2)

        wxs = []
        for i in range(self.K):
            wxs.append(torch.mm(x[i], temp[i]))
            
        wx = torch.stack(wxs)
        wx_sum = torch.sum(wx, 0)

        # Expand the mini-batch
        activation = wx_sum + self.h_bias.expand_as(wx_sum)

        # Calculate the probability p_h_given_v
        p_h_given_v = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_hid is activated or not activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        :param y: the dataset
        """

        exponents = []
        for k in range(self.K):
            wy = torch.mm(y, self.W[k])
            activation = wy + self.v_bias[k].expand_as(wy)
            exponents.append(torch.exp(activation))

        exponent_tensor = torch.stack(exponents)
        exponent_sum = torch.sum(exponent_tensor, 0)
        probs = []
        for k in range(self.K):
            p_v_k_given_h = exponent_tensor[k] / exponent_sum
            probs.append(p_v_k_given_h)

        p_v_given_h = torch.stack(probs)
        # todo multinomial
        bern = torch.bernoulli(p_v_given_h)
        return p_v_given_h, bern


    def train_model(self, v0, vk, ph0, phk):
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        ph0_K = torch.stack([ph0 for _ in range(self.K)])
        phk_K = torch.stack([phk for _ in range(self.K)])

        poss = []
        negs = []
        for i in range(self.K):
            poss.append(torch.mm(torch.transpose(v0, 1, 2)[i], ph0_K[i]))
            negs.append(torch.mm(torch.transpose(vk, 1, 2)[i], phk_K[i]))

        pos = torch.stack(poss)
        neg = torch.stack(negs)

        w_extra = torch.transpose(pos - neg, 1, 2)
        v_extra = torch.sum((v0 - vk), 1)
        h_extra = torch.sum((ph0 - phk), 0)

        # Approximate the gradients with the CD algorithm
        self.W -= self.lr() * w_extra

        # Add (difference, 0) for the tensor of 2 dimensions
        self.v_bias -= self.lr() * v_extra.unsqueeze(1)
        self.h_bias -= self.lr() * h_extra
        self.i += 1

# %%
cuda = torch.device('cuda')

# %% [markdown]
# ## General Imports

# %%
import numpy as np
import pickle as pickle
import pandas as pd
import scipy
import sklearn
import gzip
import json
from tqdm import tqdm, trange
import os
from collections import Counter
from datetime import datetime
import math
tqdm.pandas() #for progres_apply etc.

# %% [markdown]
# ## Reading in steamdata

# %%
def load_interactions(path, n_splits=5):
    """
    load in the interactions_splits.pkl.gz file with our data for the various users
    :param path: path location of the data
    :param n_splits: split in n_split splits
    :return: 
    """
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

# %% [markdown]
# ### Reading and showning train data

# %%
interactions = load_interactions("./data-cleaned/interactions_splits.pkl.gz")
interactions[0, 'train'].head()

# %% [markdown]
# ### Reading and showing list of games

# %%
games = pd.read_pickle(os.path.join(os.getcwd(), "./data-cleaned/games.pkl.gz"))
games.head()

# %%
train0 = interactions[0, 'train']
test0 = interactions[0, 'test']
val0 = interactions[0, 'val']

# %%

train1 = interactions[1, 'train']
test1 = interactions[1, 'test']
val1 = interactions[1, 'val']

# %%

train2 = interactions[2, 'train']
test2 = interactions[2, 'test']
val2 = interactions[2, 'val']

# %%


# %% [markdown]
# # Sparse Matrix

# %% [markdown]
# ### Method to create Sparse Matrix

# %%
def score_playtime(playtime):
    """
    give a game a raining score between 0 and 4 
    :param playtime: the playtime to give a score by
    :return: 0,1,2,3 or 4 based on playtime
    """
    if playtime < 120:
        # less than 2 hrs
        return 0
    elif playtime < 240:
        # less than 4 hrs
        return 1
    elif playtime < 600:
        # less than 10 hrs
        return 2
    elif playtime < 24*60:
        # less than 24 hrs
        return 3
    else:
        return 4

from os import path
if not path.isfile('index_random_ordering.pkl'):
    import random
    index_random_list =list(range(games.shape[0]))
    random.shuffle(index_random_list)
    # print(index_random_list)
    file = open("index_random_ordering.pkl", "wb")
    pickle.dump(index_random_list, file)
    file.close()
else:
    f = open("index_random_ordering.pkl", "rb")
    index_random_list = pickle.load(f)
    f.close()

def shuffle_item_index(items):
    global index_random_list
    return [index_random_list[item] for item in items]

def invert_items(items, max_item: int):
    return [max_item - item for item in items]

#Create scipy csr matrix
def get_sparse_matrix(df):
    """
    generate a sparse matrix of user-game pairs based on our dataframe
    :param df: the dataframe to base the sparse matrix upon
    :return: a sparse matrix of user-game pairs with a score based on playtime
    """
    shape = (df.shape[0], games.shape[0])
    max_game = games.shape[0] - 1
    
    user_ids = []
    item_ids = []
    values = []
    for idx, row in df.iterrows():
        user = idx
        items = row['item_id']
        # items = invert_items(items, max_game)
        items = shuffle_item_index(items)
        score = row["playtime_forever"] + 2* row["playtime_2weeks"]
        
        user_ids.extend([user] * len(items))
        item_ids.extend(items)
        values.extend([score_playtime(score[i]) for i in range(len(items))])
    # create csr matrix
    matrix = scipy.sparse.csr_matrix((values, (user_ids, item_ids)), shape=shape, dtype=np.int32)
    return matrix


# %% [markdown]
# ### generate the Sparse Matrisces for train AND test

# %%
train_matrix = get_sparse_matrix(train1)
test_matrix = get_sparse_matrix(test1)
val_matrix = get_sparse_matrix(val1)
train_matrix

# %% [markdown]
# ### Method to convert Sparse Matrix to pytorch tensor

# %%

# https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
def convert_sparse_matrix_to_sparse_tensor(X, k=5):
    """
    turn the Sparse scipy matrix into a sparse pytorch tensor
    :param X: the Sparse scipy matrix
    :param k: the amount of possible ratings we have given to our user-game pairs
    :return: a sparse 3-D pytorch tensor of dimensions game-user-rating
    """
    coo = X.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    tensor_list = []

    for index in range(k):
        value = index
        yeet = torch.where(v == value, 2., 1.)
        shape = coo.shape
        tensor = torch.sparse.DoubleTensor(i, yeet, torch.Size(shape)) 
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        tensor_list.append(tensor)

    tensor = torch.stack(tensor_list) 
    return tensor

# %% [markdown]
# # Train model

# %%
def score_model(rbm: RBM, batch_size, train_matrix, test_matrix):
    """
    calculate an error for the output of our rbm for the unseen (and untrained upon) test-values
    :param rbm: the model for which we test values
    :param batch_size: the batchsize used for training/testing
    :param train_matrix: the original input with which we try to get our test-values
    :param test_matrix: the values we try for our model to acquire based on train_matrix
    :return: the RMSE for our test-values
    """
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
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)

            # Update test RMSE reconstruction error
            test_recon_error += torch.mean((vt[vt > -1] - v[vt > -1])**2) * len(vt > -1) 
            s += len(vt > -1)

    return torch.sqrt(test_recon_error / s)


def create_rbm(train_matrix, test_matrix, n_hidden, batch_size, epochs, rbm=None, k=5, train_errors=[], test_errors=[]) -> RBM:
    """
    generate and train an RBM based on train_matrix as input
    :param train_matrix: the input upon which our model is trained
    :param test_matrix: the input upon which our model is validated
    :param n_hidden: the amount of hidden features our model uses
    :param batch_size: the batchsize we use
    :param epochs: the amount of epochs we will be running
    :param rbm: an optional variable that if not None trains a pre-generated model further instead of generating a new one
    :param k: the amount of possible ratings we have given to our user-game pairs
    :return: a trained RBM
    """
    n_vis = train_matrix.shape[1]
    if rbm is None:
        rbm = RBM(n_vis, n_hidden, k, batch_size)
    
    oldhr, oldr, oldndcg = 0, 0, 0
    # hr, r, ndcg = compute_hr(train_matrix, test_matrix, rbm)
    # print("pre training", np.average(hr), np.average(r), np.average(ndcg))

    print("start training")
    for epoch in range(epochs):
        rbm.train()
        train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
        s = 0
        
        for user_id in range(0, train_matrix.shape[0] - batch_size, batch_size):
            training_sample = train_matrix[user_id : user_id + batch_size]
            v0 = convert_sparse_matrix_to_sparse_tensor(training_sample)

            v0 = v0.to_dense()
            v0 = v0.sub(1)
            
            vk = v0.detach().clone()

            ph0, _ = rbm.sample_h(v0)
            # todo cd = 3
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
                
            phk, _ = rbm.sample_h(vk)

            rbm.train_model(v0, vk, ph0, phk)
            
            train_recon_error += torch.mean((v0[v0 > -1] - vk[v0 > -1])**2) * len(v0 > -1)
            s += len(v0 > -1)
            
        train_errors.append(torch.sqrt(train_recon_error / s))

        rbm.eval()
        test_errors.append(score_model(rbm, batch_size, train_matrix, test_matrix))

        # if epoch % 10 == 9:
        #     hr, r, ndcg = compute_hr(train_matrix, test_matrix, rbm)
        #     hr, r, ndcg = np.average(hr), np.average(r), np.average(ndcg)
        #     print(epoch, hr, r, ndcg)
        #     if ndcg < oldndcg:
        #         print("Stopping training, ndcg decreasing")
        #         break
        #     oldhr, oldr, oldndcg = hr, r, ndcg
            

    import matplotlib.pyplot as plt
    # Plot the RMSE reconstruction error with respect to increasing number of epochs
    plt.clf()
    plt.plot(torch.Tensor(train_errors, device='cpu'), label="train")
    plt.plot(torch.Tensor(test_errors, device='cpu'), label="test")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'steam-cleaned-{n_hidden}-{batch_size}-{epochs}.jpg')

    return rbm, train_errors, test_errors



# %% [markdown]
# ## HR / Recall / NDCG Function Definitions

# %% [markdown]
# ### Vanilla Recommendations

# %%
def compute_hr(train_matrix, test_matrix, rbm, k=10, rating_cutoff=-1, p=False):
    """
    compute the various metrics of our model, hr, recall and ndcg
    :param train_matrix: the input wich our user already has
    :param test_matrix: the games we are trying to recommend to each user
    :param rbm: our model used to make recommendations
    :param k: the amount of recommendations we are going to give
    :param batch_size: UNUSED, uses rbm.batch_size instead
    :return: hitrates, recall, nDCG as an array, use np.average to get value
    """
    hitrates = []
    recall = []
    nDCG = []

    recommended_games_set = set()
    # for loop - go through every single user
    # for id_user in tqdm(range(0, train_matrix.shape[0] - rbm.batch_size, rbm.batch_size)): # - batch_size, batch_size):
    for id_user in range(0, train_matrix.shape[0] - rbm.batch_size, rbm.batch_size): # - batch_size, batch_size):
        v = train_matrix[id_user:id_user + rbm.batch_size]  # training set inputs are used to activate neurons of my RBM
        vt = test_matrix[id_user:id_user + rbm.batch_size]  # target
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
        ratings, users, movies = (vt > 0).nonzero(as_tuple=True)

        indices = torch.stack([users, movies])
        shape = (rbm.batch_size, train_matrix.shape[1])
        target = torch.sparse.LongTensor(indices, torch.add(ratings, 1), torch.Size(shape))
        target_dense = target.to_dense()

        target_recommended = torch.argsort(target_dense, 1, descending=True)
        # target_rating, target_movie = torch.topk(target_dense, k, 1)
        # target_movie[target_rating < rating_cutoff] = -1 # remove all bad movies from top k

        # values, _ = torch.max(target_rating, dim=1)
        # users_with_target = (values > rating_cutoff).nonzero(as_tuple=True)[0].cpu().tolist()

        # predicted
        _, h = rbm.sample_h(v)
        recommended, _ = rbm.sample_v(h)

        scaled_tensors = [recommended[0]]
        for i in range(1, rbm.K):
            scaled_tensors.append(recommended[i] * (i+1))
        recommended_scaled = torch.stack(scaled_tensors)
        recommended_summed = torch.sum(recommended_scaled, 0)
        recommended_summed[v[0] != -1] = -10 # remove games in user lib
        predicted_rating, predicted_movie = torch.topk(recommended_summed, k)


        for user in range(rbm.batch_size):
            # all recommendations
            user_ratings = torch.index_select(target_dense[user], 0, target_recommended[user])
            user_target = target_recommended[user][user_ratings > 0].cpu().tolist()

            # user_target = target_recommended[user][target_rating[user] > rating_cutoff].cpu().tolist()
            user_pred = predicted_movie[user].cpu().tolist()

            recommended_games_set = recommended_games_set.union(set(user_pred))

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

    if p:
        print(recommended_games_set)
        print(len(recommended_games_set))
    return hitrates, recall, nDCG

# %%
# rbm10 = create_rbm(train_matrix, test_matrix, 1000, 10000, 10)

# %%
# rbm20 =create_rbm(train_matrix, test_matrix, 1000, 10000, 10, rbm10)

# %% [markdown]
# ### Popularity

# %%
def get_pop(train_df):
    popularity = train_df.explode('item_id')['item_id'].value_counts()
    # print(list(popularity.index))
    l = list(popularity.index)
    global index_random_list
    return [index_random_list[item] for item in l]
    
    # for i in range(user_reviews_df_exploded['item_id_int'].max() + 1):
    #     if i not in popularity.index:
    #         popularity[i]=0
    #     value_list.append(popularity[i])
    # # print(popularity.value.tolist())
    # print(value_list)

pop = get_pop(train0)

# %%
def compute_hr2(train_matrix, test_matrix, pops, k=10, rating_cutoff=-1, p=False, batch_size = 2000):
    hitrates = []
    recall = []
    nDCG = []
    # for loop - go through every single user
    for id_user in tqdm(range(0, train_matrix.shape[0] - batch_size, batch_size)): # - batch_size, batch_size):
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
        ratings, users, movies = (vt > 0).nonzero(as_tuple=True)

        indices = torch.stack([users, movies])
        shape = (batch_size, train_matrix.shape[1])
        target = torch.sparse.LongTensor(indices, torch.add(ratings, 1), torch.Size(shape))
        target_dense = target.to_dense()
        target_recommended = torch.argsort(target_dense, 1, descending=True)
         # remove all bad movies from top k


        # predicted
        


        for user in range(batch_size):

            # all recommendations
            user_ratings = torch.index_select(target_dense[user], 0, target_recommended[user])
            user_target = target_recommended[user][user_ratings > 0].cpu().tolist()
            user_pred = pops[:k]

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

# hr, r, ndcg = compute_hr2(train_matrix, test_matrix, pop)
# print(hr, r, ndcg)
# print("hr", np.average(hr))
# print("recall", np.average(r))
# print("ndcg", np.average(ndcg))


# %% [markdown]
# ### RBM + Popularity

# %%
def compute_hr3(rbm, popularity_dict, k=10):
    """
    
    :param rbm: 
    :param popularity_dict: 
    :param k: 
    :return: 
    """

    # return hitrates, recall, nDCG
    pass

# %% [markdown]
# ## calculate HR , Recall & NDCG for our RBM

# %%
def evaluate_rbm(rbm, train_matrix, test_matrix, p=True):
    print("Vanilla RBM")
    hr, r, ndcg = compute_hr(train_matrix, test_matrix, rbm, p=p)
    # print(hr, r, ndcg)
    print("hr", np.average(hr))
    print("recall", np.average(r))
    print("ndcg", np.average(ndcg))


# %%
rbms = []

# %%
for i in range(5):
    rbm = create_rbm(train_matrix, val_matrix, 5, 2000, 100, train_errors=[], test_errors=[])[0]
    print(f"RBM {i}")
    evaluate_rbm(rbm, train_matrix, test_matrix)
    with open(f"./rbm5-cedric-100-nr{i}.pickle", "wb") as f:
        pickle.dump(rbm, f)
    # torch.save(rbm.state_dict(), f"./rbm5-run2-100-nr{i}.network")
    rbms.append(rbm)


# %% [markdown]
# # recommend for single user

# %% [markdown]
# # Ensemble of ANTI-RBM

# %%
def compute_ensemble_hr(train_matrix, test_matrix, rbms: list[RBM], k=10, rating_cutoff=-1, p=False):
    """
    compute the various metrics of our model, hr, recall and ndcg
    :param train_matrix: the input wich our user already has
    :param test_matrix: the games we are trying to recommend to each user
    :param rbm: our model used to make recommendations
    :param k: the amount of recommendations we are going to give
    :param batch_size: UNUSED, uses rbm.batch_size instead
    :return: hitrates, recall, nDCG as an array, use np.average to get value
    """
    hitrates = []
    recall = []
    nDCG = []
    batch_size = rbms[0].batch_size

    recommended_games_set = set()
    # for loop - go through every single user
    # for id_user in tqdm(range(0, train_matrix.shape[0] - rbm.batch_size, rbm.batch_size)): # - batch_size, batch_size):
    for id_user in trange(0, train_matrix.shape[0] - batch_size, batch_size): # - batch_size, batch_size):
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
        ratings, users, movies = (vt > 0).nonzero(as_tuple=True)

        indices = torch.stack([users, movies])
        shape = (batch_size, train_matrix.shape[1])
        target = torch.sparse.LongTensor(indices, torch.add(ratings, 1), torch.Size(shape))
        target_dense = target.to_dense()

        target_recommended = torch.argsort(target_dense, 1, descending=True)

        # predicted
        pred_ratings = []
        pred_movies = []
        for rbm in rbms:
            _, h = rbm.sample_h(v)
            recommended, _ = rbm.sample_v(h)

            scaled_tensors = [recommended[0]]
            for i in range(1, rbm.K):
                scaled_tensors.append(recommended[i] * (i+1))
            recommended_scaled = torch.stack(scaled_tensors)
            recommended_summed = torch.sum(recommended_scaled, 0)
            pred_rating, pred_movie = torch.topk(recommended_summed, k)
            pred_ratings.append(pred_rating)
            pred_movies.append(pred_movie)

        predicted_ratings = torch.cat(pred_ratings, dim=1)
        predicted_movies = torch.cat(pred_movies, dim=1)
        _, predicted_indices = torch.topk(predicted_ratings, k)


        for user in range(batch_size):
            # all recommendations
            user_ratings = torch.index_select(target_dense[user], 0, target_recommended[user])
            user_target = target_recommended[user][user_ratings > 0].cpu().tolist()

            # user_target = target_recommended[user][target_rating[user] > rating_cutoff].cpu().tolist()
            user_pred = torch.index_select(predicted_movies[user], 0, predicted_indices[0]).cpu().tolist()

            recommended_games_set = recommended_games_set.union(set(user_pred))
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

    if p:
        print(recommended_games_set)
        print(len(recommended_games_set))
    return hitrates, recall, nDCG

# %%
def compute_ensemble_hr2(train_matrix, test_matrix, rbms: list[RBM], k=10, rating_cutoff=-1, p=False):
    """
    compute the various metrics of our model, hr, recall and ndcg
    :param train_matrix: the input wich our user already has
    :param test_matrix: the games we are trying to recommend to each user
    :param rbm: our model used to make recommendations
    :param k: the amount of recommendations we are going to give
    :param batch_size: UNUSED, uses rbm.batch_size instead
    :return: hitrates, recall, nDCG as an array, use np.average to get value
    """
    hitrates = []
    recall = []
    nDCG = []
    batch_size = rbms[0].batch_size

    recommended_games_set = set()
    # for loop - go through every single user
    # for id_user in tqdm(range(0, train_matrix.shape[0] - rbm.batch_size, rbm.batch_size)): # - batch_size, batch_size):
    for id_user in trange(0, train_matrix.shape[0] - batch_size, batch_size): # - batch_size, batch_size):
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
        ratings, users, movies = (vt > 0).nonzero(as_tuple=True)

        indices = torch.stack([users, movies])
        shape = (batch_size, train_matrix.shape[1])
        target = torch.sparse.LongTensor(indices, torch.add(ratings, 1), torch.Size(shape))
        target_dense = target.to_dense()

        target_recommended = torch.argsort(target_dense, 1, descending=True)

        # predicted
        pred_ratings = []
        pred_movies = []
        for rbm in rbms:
            _, h = rbm.sample_h(v)
            recommended, _ = rbm.sample_v(h)

            scaled_tensors = [recommended[0]]
            for i in range(1, rbm.K):
                scaled_tensors.append(recommended[i] * (i+1))
            recommended_scaled = torch.stack(scaled_tensors)
            recommended_summed = torch.sum(recommended_scaled, 0)
            pred_rating, pred_movie = torch.topk(recommended_summed, k // len(rbms))
            pred_ratings.append(pred_rating)
            pred_movies.append(pred_movie)

        predicted_movies = torch.cat(pred_movies, dim=1)

        for user in range(batch_size):
            # all recommendations
            user_ratings = torch.index_select(target_dense[user], 0, target_recommended[user])
            user_target = target_recommended[user][user_ratings > 0].cpu().tolist()

            user_pred = predicted_movies[user].cpu().tolist()

            recommended_games_set = recommended_games_set.union(set(user_pred))
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

    if p:
        print(recommended_games_set)
        print(len(recommended_games_set))
    return hitrates, recall, nDCG


# %%
for k in [5, 10, 20]:
    hr, r, ndcg = compute_ensemble_hr(train_matrix, test_matrix, rbms, k=k)
    print("k:", k)
    print(np.average(hr), np.average(r), np.average(ndcg))

# %%
for k in [5, 10, 20]:
    hr, r, ndcg = compute_ensemble_hr2(train_matrix, test_matrix, rbms, p=True, k=k)
    print("k:", k)
    print(np.average(hr), np.average(r), np.average(ndcg))

# %%
metrics_dict = {}
for rbm in rbms:
    print("---------")
    for k in [5, 10, 20]:
        hr, r, ndcg = compute_hr(train_matrix, test_matrix, rbm, k=k)
        print("k:", k)
        print(np.average(hr), np.average(r), np.average(ndcg))

        if ("hr", k) not in metrics_dict:
            metrics_dict["hr", k] = [np.average(hr)]
            metrics_dict["r", k] = [np.average(r)]
            metrics_dict["ndcg", k] = [np.average(ndcg)]
        else:
            metrics_dict["hr", k] += [np.average(hr)]
            metrics_dict["r", k] += [np.average(r)]
            metrics_dict["ndcg", k] += [np.average(ndcg)]

for key, value in metrics_dict.items():
    print(key, np.average(value))


