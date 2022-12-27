import torch
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
import random
from torch.linalg import det

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=True):
    nx = x.shape[0]
    ny = y.shape[0]

    M = pdist(x, y)
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a
    u = a
    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()
    E = upper_t * Mt
    D = 2 * torch.sum(E)
    if cuda:
        D = D.to(device)

    return D

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))

def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)

def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result

def invmat(M):
    return 1.0/det(M)*adj(M)

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def wasserstein_cost_loss(output, sens):

    group_1_diff = output[sens==1]
    group_0_diff = output[sens==0]

    return wasserstein(group_0_diff, group_1_diff, 'cuda')

def grad_z_graph_faircost(adj, features, idx, labels, sens, model, gpu=-1):
    model.eval()
    # initialize
    if gpu >= 0:
        adj, features, labels = adj.cuda(), features.cuda(), labels.cuda()
    if gpu < 0:
        adj, features, labels, model = adj.cpu(), features.cpu(), labels.cpu(), model.cpu()

    output = model(features, adj)

    if output.shape[1] == 1:
        # SP
        # loss = wasserstein_cost_loss(output[idx], sens[idx])

        # EO
        # loss = wasserstein_cost_loss(output[idx][labels[idx] == 1], sens[idx][labels[idx] == 1])

        # SP + EO
        loss1 = wasserstein_cost_loss(output[idx], sens[idx])
        loss2 = wasserstein_cost_loss(output[idx][labels[idx] == 1], sens[idx][labels[idx] == 1])
        loss = 0.5 * loss1 + 0.5 * loss2

    else:
        assert 0 == 1

    params = [p for p in model.parameters() if p.requires_grad]

    return list(grad(loss, params, create_graph=False))


def s_test_graph_cost(adj, features, idx_train, idx_test, labels, sens, model, gpu=-1, damp=0.03, scale=60,
           recursion_depth=5000):
    # For Pokec2 with GCN:
    # setting scale as 50 to achieve better estimated Pearson correlation;
    # setting scale as 60 to achieve better GNN debiasing performance.

    v = grad_z_graph_faircost(adj, features, idx_test, labels, sens, model, gpu)
    h_estimate_cost = v.copy()

    if gpu >= 0:
        adj, features, labels = adj.cuda(), features.cuda(), labels.cuda()
    if gpu < 0:
        adj, features, labels, model = adj.cpu(), features.cpu(), labels.cpu(), model.cpu()

    for i in range(recursion_depth):
        random_train_idx = idx_train[random.randint(0, len(idx_train) - 1)]
        output = model(features, adj)
        out, label = output[random_train_idx], labels[random_train_idx].unsqueeze(0).float()
        loss = F.binary_cross_entropy_with_logits(out, label)
        params = [p for p in model.parameters() if p.requires_grad]
        hv = hvp(loss, params, h_estimate_cost)
        h_estimate_cost = [
            _v + (1 - damp) * _h_e - _hv / scale
            for _v, _h_e, _hv in zip(v, h_estimate_cost, hv)]

    h_estimate_cost = [b / scale for b in h_estimate_cost]
    return h_estimate_cost

def grad_z_graph(adj, features, idx, labels, model, gpu=-1):

    model.eval()
    if gpu >= 0:
        adj, features, labels = adj.cuda(), features.cuda(), labels.cuda()
    if gpu < 0:
        adj, features, labels, model = adj.cpu(), features.cpu(), labels.cpu(), model.cpu()

    gradients_list = []
    for i in range(len(idx)):
        output = model(features, adj)
        out, label = output[idx[i]], labels[idx[i]].unsqueeze(0).float()
        loss = F.binary_cross_entropy_with_logits(out, label)
        params = [p for p in model.parameters() if p.requires_grad]
        gradients_list.append(list(grad(loss, params, create_graph=False)))

    return gradients_list

def hvp(y, w, v):

    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    return_grads = grad(elemwise_products, w, create_graph=False)

    return return_grads

def cal_influence_graph(idx_train, h_estimate_mean, gradients_list, gpu=0):

    assert len(gradients_list) == len(idx_train)
    grad_z_vecs = gradients_list
    e_s_test = h_estimate_mean

    influences = []
    for i in range(len(idx_train)):
        tmp_influence = -sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
            ]) / len(idx_train)
        influences.append(tmp_influence)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, idx_train[harmful.copy()].tolist(), idx_train[helpful.copy()].tolist(), harmful.tolist(), helpful.tolist()

def cal_influence_graph_nodal(idx_all, idx_train, h_estimate_mean, gradients_list, gpu=0):

    assert len(gradients_list) == len(idx_train)
    grad_z_vecs = gradients_list
    e_s_test = h_estimate_mean
    influences = []
    for i in range(len(idx_train)):
        tmp_influence = -sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
            ]) / (len(idx_all))
        influences.append(tmp_influence)
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, idx_train[harmful.copy()].tolist(), idx_train[helpful.copy()].tolist(), harmful.tolist(), helpful.tolist()