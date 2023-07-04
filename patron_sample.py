import re
import torch
import numpy as np
from collections import Counter
import time
from datetime import datetime 

from scipy import stats
import pandas as pd
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import faiss
from tqdm import tqdm, trange 
from sklearn.metrics import pairwise_distances
import copy 
import pickle
import json
import argparse
import math 
import os
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use("agg")


def patron(unlabel_value, unlabeled_feat, n_sample, k = 50, rho = 0.1, beta = 0.1, mu = 0.1, gamma = 0.5, refine_round = 3, prop = True):
    '''
        unlabeled_value: the uncertainty value estimated from the Eq. (5) of the paper 
        unlabeled_feat: the embeddings of the unlabeled data
        n_sample: the total number of samples used for acquisition |B|
        k: the parameter used for KNN calculation for uncertainty propagation
        beta: the regularization of distance 
        mu: the margin
        gamma: the weight of the  regularization term in Eq. (10)
    '''
    d = unlabeled_feat.shape[-1] 
    n_data = unlabeled_feat.shape[0] 
    if prop: 
        index = faiss.IndexFlatL2(d) 
        index.add(unlabeled_feat) 
        D, I = index.search(unlabeled_feat, k+1) 
        prop_score =  np.mean(unlabel_value[I] * np.exp(-D*rho), axis = -1)
    else:
        prop_score = unlabel_value
    kmeans = faiss.Kmeans(d, n_sample, niter=100, verbose=True, nredo = 5)
    kmeans.train(unlabeled_feat)
    D, I = kmeans.index.search(unlabeled_feat, 1)
    cluster_id = I.flatten()
    sample_idx_rounds = []
    sample_idx = []
    visited = {}
    for i in range(n_sample):
        idxs_i = np.arange(n_data)[cluster_id == i]
        typi_i = prop_score[cluster_id == i]
        feat_i = unlabeled_feat[cluster_id == i]
        dist_i = np.linalg.norm(feat_i - kmeans.centroids[i], axis = -1)
        index = idxs_i[np.argmax(typi_i -  beta * dist_i)]
        sample_idx.append(int(index))
        visited[index] = 1
    sample_idx_rounds.append(sample_idx)
    print(f"beta:{beta}, mu:{mu}, gamma:{gamma}, Round 0: {sample_idx}")
    for refine_i in range(refine_round):
        sample_idx = []
        visited = {}
        prev_centers = unlabeled_feat[sample_idx_rounds[-1]]
        # (n_sample, 768)
        index = faiss.IndexFlatL2(d) 
        index.add(prev_centers) 
        D, I = kmeans.index.search(unlabeled_feat, 11)
        D = D[:, 1:]
        cluster_id = I[:, 0].flatten()
        for i in range(n_sample):
            idxs_i = np.arange(n_data)[cluster_id == i]
            typi_i = prop_score[cluster_id == i]
            feat_i = unlabeled_feat[cluster_id == i]
            dist_i = np.linalg.norm(feat_i - np.mean(feat_i, axis = 0, keepdims=True), axis = -1)
            dist_to_near_nei = D[cluster_id == i]                
            dist_to_near_nei = np.clip(dist_to_near_nei, a_max = mu, a_min = 0)
            dist_to_near_nei = np.mean(dist_to_near_nei, axis = -1)
            index = idxs_i[np.argmax(typi_i -  beta * dist_i + gamma * dist_to_near_nei)]
            sample_idx.append(int(index))
            visited[index] = 1
        print(f"beta:{beta}, mu:{mu}, gamma:{gamma}, Round {refine_i + 1}: {sample_idx}")
        sample_idx_rounds.append(sample_idx)
    return sample_idx_rounds

''' loading embedding and predictions '''
def load_data(dataset = 'IMDB', embedding_model = 'roberta-base', template_id = 0):
    path = f'{dataset}/'
    with open(path + f'embedding_{embedding_model}_roberta.pkl', 'rb') as f:
        train_emb = pickle.load(f)    
    train_prompt_pred = np.load(path + f"pred_unlabeled_roberta-base_temp{template_id}.npy")
    # train_label = np.load(path + "pred_labels.npy") # actually unused

    # assert len(test_label) == test_emb.shape[0]
    assert train_emb.shape[0] == train_label.shape[0]
    assert train_emb.shape[0] == train_prompt_pred.shape[0]
    return train_emb, train_prompt_pred


''' loading training data '''
def load_id(method = 'badge', dataset = 'agnews', nlabel = 16, model = 'roberta-base'):
    path = f'{dataset}/'
    train_name = path + f'train_idx_{model}_{method}_{nlabel}.json'
    with open(train_name, 'r') as f:
        train_idx = json.load(f)
    train_idx = np.array(train_idx, dtype = int)
    return train_idx


''' loading training arguments '''
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='agnews',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--model",
        default='simcse',
        type=str,
        help="The model used for embedding",
    )

    parser.add_argument(
        "--template",
        default=0,
        type=int,
        help="The template id for prompts",
    )

    parser.add_argument(
        "--prop",
        default=1,
        type=int,
        help="Whether use uncertainty propagation or not",
    )

    parser.add_argument(
        "--k",
        default=50,
        type=int,
        help="The size of the neighborhood size",
    )

    parser.add_argument(
        "--rho",
        default=0.01,
        type=float,
        help="The weight for controlling the propagation in Eq. (6)",
    )

    parser.add_argument(
        "--gamma",
        default=0.3,
        type=float,
        help="The weight of the  regularization term in Eq. (10)",
    )

    parser.add_argument(
        "--beta",
        default=1,
        type=float,
        help="The weight of the balancing term in Eq. (8)",
    )

    parser.add_argument(
        "--mu",
        default=0.5,
        type=float,
        help="The margin of the  regularization term in Eq. (10)",
    )

    parser.add_argument(
        "--n_sample",
        default=32,
        type=int,
        help="The number of acquired data size",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    Suppose all the data is in the folder ./X, where X = {AGNews, IMDB, TREC, Yahoo, Yelp-full}
    '''
    args = get_arguments()

    prop = args.prop
    k = args.k
    rho = args.rho
    beta = args.beta
    mu = args.mu
    gamma = args.gamma
    n_sample = args.n_sample

    print(f"Using, Prop: {prop}")
    train_emb, train_prompt_pred = load_data(args.dataset, args.model, template_id = args.template)
    mean_pred = np.mean(train_prompt_pred, axis =  0)

    train_prompt_pred = train_prompt_pred / np.sum(train_prompt_pred, axis=-1, keepdims= True)
    entropy = np.sum(-np.log(train_prompt_pred + 1e-12) * train_prompt_pred, axis = -1)

    ##########
    local_uncertainty = entropy.flatten()

    sample_idxs = patron(local_uncertainty, train_emb, n_sample = n_sample, k = k, rho = rho, beta = beta, mu = mu, gamma = gamma,  refine_round = 1, prop = prop)
    for round, sample_idx in enumerate(sample_idxs):
        with open(f"{args.dataset}/train_idx_roberta-base_round{round}_rho{args.rho}_gamma{gamma}_beta{beta}_mu{mu}_{n_sample}.json", 'w') as f:
            json.dump(sample_idx, f)
            
