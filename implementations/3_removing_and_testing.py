from __future__ import division
from __future__ import print_function


from implementations.approximator import wasserstein_cost_loss
from numpy import *
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance_matrix
import os
import networkx as nx
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance
from implementations.utils import load_bail, load_income, load_pokec_renewed
import warnings
warnings.filterwarnings('ignore')
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default="bail", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to continue training for estimation evaluation.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--helpfulness_collection', type=int, default=0,
                    help='do leave-one-out for helpful nodes.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
dataset_name = args.dataset
open_factor = args.helpfulness_collection
print(open_factor)

def accuracy_new(output, labels):
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)

    return idx_map

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adj(dataset_name):
    predict_attr = "RECID"
    if dataset_name == 'bail':
        predict_attr="RECID"
    elif dataset_name == 'income':
        predict_attr = "income"

    if dataset_name == 'pokec1' or dataset_name == 'pokec2':
        if dataset_name == 'pokec1':
            edges = np.load('../data/pokec_dataset/region_job_1_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_1_labels.npy')
        else:
            edges = np.load('../data/pokec_dataset/region_job_2_2_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_2_2_labels.npy')

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj

    path="../data/" + str(dataset_name) + "/"
    dataset = dataset_name
    print('Reconstructing the adj of {} dataset...'.format(dataset))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj

def del_adj(harmful, dataset_name):
    predict_attr = "RECID"

    if dataset_name == 'bail':
        predict_attr="RECID"
    elif dataset_name == 'income':
        predict_attr = "income"

    if dataset_name == 'pokec1' or dataset_name == 'pokec2':
        if dataset_name == 'pokec1':
            edges = np.load('../data/pokec_dataset/region_job_1_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_1_labels.npy')
        else:
            edges = np.load('../data/pokec_dataset/region_job_2_2_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_2_2_labels.npy')

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        mask = np.ones(adj.shape[0], dtype=bool)
        mask[harmful] = False
        adj = sp.coo_matrix(adj.tocsr()[mask, :][:, mask])
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        return adj

    path="../data/" + str(dataset_name) + "/"
    dataset = dataset_name
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    adj = adj_ori
    mask = np.ones(adj.shape[0], dtype=bool)
    mask[harmful] = False
    adj = sp.coo_matrix(adj.tocsr()[mask,:][:,mask])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    return adj

def find123Nei(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):
        nei1_li .append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)

    return [nei1_li, nei2_li, nei3_li]

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

if dataset_name == 'bail':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail')
    norm_features = feature_norm(features)
    norm_features[:, 0] = features[:, 0]
    features = feature_norm(features)
elif dataset_name == 'income':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_income('income')
    norm_features = feature_norm(features)
    norm_features[:, 8] = features[:, 8]
    features = feature_norm(features)
elif dataset_name == 'pokec1':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(1)
elif dataset_name == 'pokec2':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(2)

edge_index = convert.from_scipy_sparse_matrix(adj)[0]
computation_graph_involving = []
the_adj = get_adj(dataset_name)
hop = 1
print("Finding neighbors ... ")
G = nx.Graph(the_adj)
for i in tqdm(range(idx_train.shape[0])):
    neighbors = find123Nei(G, idx_train[i].item())
    mid = []
    for j in range(hop):
        mid += neighbors[j]
    mid = list(set(mid).intersection(set(idx_train.numpy().tolist()))) + [idx_train[i].item()]
    computation_graph_involving.append(mid)

final_influence = np.load('final_influence_' + dataset_name + '.npy', allow_pickle=True)
helpful = idx_train[np.argsort(final_influence).copy()].tolist()
helpful_idx = np.argsort(final_influence).copy().tolist()
harmful_idx = helpful_idx[::-1]
harmful = idx_train[harmful_idx].tolist()

if open_factor:
    harmful = helpful
    harmful_idx = helpful_idx

total_neighbors = []
masker = np.ones(len(harmful), dtype=bool)
for i in range(len(harmful) - 1):
    if masker[i] == True:
        total_neighbors += computation_graph_involving[harmful_idx[i]]
    if list(set(total_neighbors).intersection(set(computation_graph_involving[harmful_idx[i + 1]]))) != []:  # != [] 找的是nonoverlapping的
        masker[i+1] = False

harmful_idx = np.array(harmful_idx)[masker].tolist()
harmful = idx_train[harmful_idx].tolist()

max_num = 0
for i in range(len(final_influence[harmful_idx]) - 1):
    if final_influence[harmful_idx][i] * final_influence[harmful_idx][i+1] <= 0:
        print("At most effective number:")
        print(i + 1)
        max_num = i + 1
        break

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

    return parity.item(), equality.item()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    preds = (output.squeeze() > 0).type_as(labels)
    acc_train = accuracy_new(preds[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, edge_index)

    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    acc_val = accuracy_new(preds[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    # print(wasserstein_cost_loss(output[idx_test], sens[idx_test]))

    return loss_val.item()


def tst():
    model.eval()
    output = model(features, edge_index)
    preds = (output.squeeze() > 0).type_as(labels)
    loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
    acc_test = accuracy_new(preds[idx_test], labels[idx_test])

    idx_sens_test = sens[idx_test]
    idx_output_test = output[idx_test]
    fair_cost_records.append(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

    auc_roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
    f1_test = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
    parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                   sens[idx_test].numpy())

    sp_records.append(parity)
    eo_records.append(equality)
    acc_records.append(acc_test.item())
    auc_records.append(auc_roc_test)
    f1_records.append(f1_test)


if dataset_name == 'bail':
    adj_ori, features_ori, labels_ori, idx_train_ori, idx_val_ori, idx_test_ori, sens_ori = load_bail('bail')
    norm_features_ori = feature_norm(features_ori)
    norm_features_ori[:, 0] = features_ori[:, 0]
    features_ori = feature_norm(features_ori)
elif dataset_name == 'income':
    adj_ori, features_ori, labels_ori, idx_train_ori, idx_val_ori, idx_test_ori, sens_ori = load_income('income')
    norm_features_ori = feature_norm(features_ori)
    norm_features_ori[:, 8] = features_ori[:, 8]
    features_ori = feature_norm(features_ori)
elif dataset_name == 'pokec1':
    adj_ori, features_ori, labels_ori, idx_train_ori, idx_val_ori, idx_test_ori, sens_ori = load_pokec_renewed(1)
elif dataset_name == 'pokec2':
    adj_ori, features_ori, labels_ori, idx_train_ori, idx_val_ori, idx_test_ori, sens_ori = load_pokec_renewed(2)

edge_index_ori = convert.from_scipy_sparse_matrix(adj_ori)[0]

influence_approximation = []
fair_cost_records = []
sp_records = []
eo_records = []
acc_records = []
auc_records = []
f1_records = []

batch_size = 1
percetage_budget = 0.3


for num_of_deleting in tqdm(range(int(percetage_budget * max_num//batch_size) + 1)):

    adj, features, labels, idx_train, idx_val, idx_test, sens = adj_ori, features_ori.clone(), labels_ori.clone(), idx_train_ori.clone(), idx_val_ori.clone(), idx_test_ori.clone(), sens_ori.clone()
    bin = num_of_deleting
    k = int(batch_size * bin)
    harmful_flags = harmful[:k]

    influence_approximation.append(sum(final_influence[harmful_idx[:k]]))
    harmful_idx_flags = harmful_idx[:k]
    mask = np.ones(idx_train.shape[0], dtype=bool)
    mask[harmful_idx_flags] = False
    idx_train = idx_train[mask]
    idx_val = idx_val.clone()
    idx_test = idx_test.clone()

    reference = list(range(adj.shape[0]))
    for i in range(len(harmful_flags)):
        for j in range(len(reference) - harmful_flags[i]):
            reference[j + harmful_flags[i]] -= 1

    idx_train = torch.LongTensor(np.array(reference)[idx_train.numpy()])
    idx_val = torch.LongTensor(np.array(reference)[idx_val.numpy()])
    idx_test = torch.LongTensor(np.array(reference)[idx_test.numpy()])

    mask = np.ones(labels.shape[0], dtype=bool)
    mask[harmful_flags] = False
    features = features[mask, :]
    labels = labels[mask]
    sens = sens[mask]

    adj = del_adj(harmful_flags, dataset_name)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    model = torch.load('gcn_' + dataset_name + '.pth')
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        edge_index = edge_index.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    final_epochs = -1
    loss_val_global = 1e10

    for epoch in range(args.epochs):
        loss_mid = train(epoch)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'mid_best' + str(bin) + '.pth')
            final_epochs = epoch

    if final_epochs == -1:
        assert 1 == 0

    model = torch.load('mid_best' + str(bin) + '.pth')
    tst()
    os.remove('mid_best' + str(bin) + '.pth')

final_sets = {}
final_sets['influence_approximation'] = influence_approximation
final_sets['fair_cost_records'] = fair_cost_records
final_sets['sp_records'] = sp_records
final_sets['eo_records'] = eo_records
final_sets['acc_records'] = acc_records
final_sets['auc_records'] = auc_records
final_sets['f1_records'] = f1_records

if open_factor:
    np.save('1final_sets_' + dataset_name + '.npy', final_sets)
else:
    np.save('imp_final_sets_' + dataset_name + '.npy', final_sets)
