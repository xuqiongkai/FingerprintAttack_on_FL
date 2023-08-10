#!/usr/bin/env python3
# Copyright (c)
#        Qiongkai Xu
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from sklearn.cluster import KMeans,SpectralClustering

from sklearn import cluster
from sklearn import metrics
import numpy as np

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import utils.analyse_utils as analyse_utils
import argparse

exp_dir = '~/exp/LM-fingerprint/news/local-gpt2-xxx'
# exp_dir = '~/exp/LM-fingerprint/dial/local-gpt2-xxx'


print(exp_dir)

def parsee_args():
    parser = argparse.ArgumentParser(description="Gradient Analysis for Fingerprinting.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--cluster", default='kmean', help="Clustering Algorithm (kmean/spectral/greedy_match/)")
    parser.add_argument("--eval", default='simple', help="Evaluation metrics (simple/all)")
    parser.add_argument("--epochs", default=10, type=int, help="Total epoches considered in test.")
    parser.add_argument("--clients", default=20, type=int, help="Total clients considered in test.")
    
    args = parser.parse_args()
    return args

args = parsee_args()
print(args)


params = []
labels = []

weight_name = 'transformers.0'


def extract_raw_feature(grads, module_name):
    param = [weight[:,:].numpy().flatten() for name, weight in grads if module_name in name and ('ff.fc.weight' in name or 'ff.proj.weight' in name)]
    return param


def extract_features(grads, client, batch_id):
    param = extract_raw_feature(grads, weight_name)
    
    param = np.concatenate(param)
    params.append(param)
    labels.append(client)


for c in range(args.clients):
    for b in range(args.epochs):
        d = torch.load(os.path.join(exp_dir, 'client_{}_e{}.pt'.format(c, b+1)))
        p = extract_features(d, c, b)


### refer to https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": args.clients,
    "min_samples": args.epochs,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

### setup params
 # normalize dataset for easier parameter selection
params = StandardScaler().fit_transform(params)

# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(params, quantile=default_base["quantile"])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
    params, n_neighbors=default_base["n_neighbors"], include_self=False
)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)


if args.cluster == 'kmean':
    cluster = KMeans(n_clusters=args.clients, random_state=42) 


elif args.cluster == 'spectral':
    cluster = SpectralClustering(n_clusters=args.clients,
            eigen_solver="arpack",
            affinity="nearest_neighbors")

elif args.cluster == 'greedy_match':
    cluster = analyse_utils.PairwiseClustering(args.clients, args.epochs)


cluster.fit(params)
res = np.array(cluster.labels_)
gt = np.array(labels)


PUR = analyse_utils.purity_score(gt, res)
RI = metrics.rand_score(gt, res)
MI = metrics.mutual_info_score(gt, res)

if args.eval == 'all':
    ARI = metrics.adjusted_rand_score(gt, res)
    AMI = metrics.adjusted_mutual_info_score(gt, res)

if args.eval == 'simple':
    print('#', weight_name, args.epochs, args.cluster)
    print('e {} c {}: [ {:.3f} , {:.3f} , {:.3f} ],'.format(args.epochs, args.clients, PUR, RI, MI))
elif args.eval == 'all':
    print('#', weight_name, args.epochs, args.cluster)
    print('e {} c {}: [ {:.3f} , {:.3f} , {:.3f} , {:.3f} , {:.3f} ]'.format(args.epochs, args.clients, PUR, RI, ARI, MI, AMI))
    
