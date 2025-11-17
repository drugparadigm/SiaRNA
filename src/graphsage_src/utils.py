import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

def get_gnn_embeddings(gnn_model, dataCenter, ds, adj_lists):
    test_nodes = getattr(dataCenter, ds+'_test')
    train_nodes = getattr(dataCenter, ds+'_train')
    val_nodes = getattr(dataCenter, ds+'_val')
    nodes = np.concatenate((test_nodes, train_nodes, val_nodes))
    nodes = np.unique(nodes)
    nodes = np.sort(nodes)
    # print(test_nodes)
    # print(type(test_nodes))
    for i in test_nodes:
        if i not in adj_lists:
            adj_lists[i].add(i)
    #print(adj_lists)
    # print(nodesa)
    # nodesb = getattr(dataCenter, ds+'_train')
    # print(nodesb)
    # # 找到在 a 中但不在 b 中的元素
    # result = np.setdiff1d(nodesa, nodesb)
    # print(result)
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        # print(f'打印nodes_batch: {nodes_batch}')
        # print(nodes_batch)
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    # print(embs.size())
    assert len(embs) == len(nodes)
    return embs.detach()

def get_unseen_gnn_embeddings(gnn_model, dataCenter, ds, adj_lists):

    unseen_nodes = getattr(dataCenter, ds+'_unseen')

    # print(test_nodes)
    # print(type(test_nodes))
    # Add the ID of a node that is not in the training set
    for i in unseen_nodes:
        if i not in adj_lists:
            adj_lists[i].add(i)
            # print(i)
    #print(adj_lists)
    # print(nodesa)
    # nodesb = getattr(dataCenter, ds+'_train')
    # print(nodesb)
    # # 找到在 a 中但不在 b 中的元素
    # result = np.setdiff1d(nodesa, nodesb)
    # print(result)
    b_sz = 200
    batches = math.ceil(len(unseen_nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = unseen_nodes[index*b_sz:(index+1)*b_sz]
        # print(f'打印nodes_batch: {nodes_batch}')
        # print(nodes_batch)
        embs_batch = gnn_model(nodes_batch)
        # print(embs_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    print(embs.size())
    assert len(embs) == len(unseen_nodes)
    return embs.detach()

# def get_gnn_embeddings(gnn_model, dataCenter, ds):
#     print('Loading embeddings from trained GraphSAGE model.')
#     features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
#     nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
#     b_sz = 500
#     batches = math.ceil(len(nodes) / b_sz)
#     embs = []
#     for index in range(batches):
#         nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
#         embs_batch = gnn_model(nodes_batch)
#         assert len(embs_batch) == len(nodes_batch)
#         embs.append(embs_batch)
#         # if ((index+1)*b_sz) % 10000 == 0:
#         #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')
#
#     assert len(embs) == batches
#     embs = torch.cat(embs, 0)
#     assert len(embs) == len(nodes)
#     print('Embeddings loaded.')
#     return embs.detach()

# def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
#     test_nodes = getattr(dataCenter, ds+'_test')
#     # print(test_nodes)
#     val_nodes = getattr(dataCenter, ds+'_val')
#     train_nodes = getattr(dataCenter, ds+'_train')
#     # print(train_nodes)
#     intersection = list(set(test_nodes) & set(train_nodes))
#
#     # print("两个列表的交集是:", intersection)
#     labels = getattr(dataCenter, ds+'_labels')
#
#     if unsup_loss == 'margin':
#         num_neg = 6
#     elif unsup_loss == 'normal':
#         num_neg = 100
#     else:
#         print("unsup_loss can be only 'margin' or 'normal'.")
#         sys.exit(1)
#
#     train_nodes = shuffle(train_nodes)
#
#     models = [graphSage, classification]
#     params = []
#     for model in models:
#         for param in model.parameters():
#             if param.requires_grad:
#                 params.append(param)
#     #print(params)
#     optimizer = torch.optim.SGD(params, lr=0.1)
#     optimizer.zero_grad()
#     for model in models:
#         model.zero_grad()
#
#     batches = math.ceil(len(train_nodes) / b_sz)
#
#     visited_nodes = set()
#     for index in range(batches):
#         nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
#         #print(nodes_batch)
#         # extend nodes batch for unspervised learning
#         # no conflicts with supervised learning
#         nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
#         #print(len(nodes_batch))
#         visited_nodes |= set(nodes_batch)
#
#         # print(visited_nodes)
#
#         # get ground-truth for the nodes batch
#         labels_batch = labels[nodes_batch]
#
#         # feed nodes batch to the graphSAGE
#         # returning the nodes embeddings
#         embs_batch = graphSage(nodes_batch)
#         #print(embs_batch)
#
#         if learn_method == 'sup':
#             # superivsed learning
#             logists = classification(embs_batch)
#             loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
#             loss_sup /= len(nodes_batch)
#             loss = loss_sup
#         elif learn_method == 'plus_unsup':
#             # superivsed learning
#             logists = classification(embs_batch)
#             loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
#             loss_sup /= len(nodes_batch)
#             # unsuperivsed learning
#             if unsup_loss == 'margin':
#                 loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
#             elif unsup_loss == 'normal':
#                 loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
#             loss = loss_sup + loss_net
#         else:
#             if unsup_loss == 'margin':
#                 loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
#             elif unsup_loss == 'normal':
#                 loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
#             loss = loss_net
#             #print(embs_batch)
#
#         print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
#         loss.backward()
#         for model in models:
#             nn.utils.clip_grad_norm_(model.parameters(), 5)
#         optimizer.step()
#
#         optimizer.zero_grad()
#         for model in models:
#             model.zero_grad()
#
#     return graphSage, classification

def apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
    test_nodes = getattr(dataCenter, ds+'_test')
    # print(test_nodes)
    val_nodes = getattr(dataCenter, ds+'_val')
    train_nodes = getattr(dataCenter, ds+'_train')
    # print(train_nodes)
    intersection = list(set(test_nodes) & set(train_nodes))

    # print("两个列表的交集是:", intersection)
    # labels = getattr(dataCenter, ds+'_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    #print(params)
    optimizer = torch.optim.SGD(params, lr=0.65)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
        #print(nodes_batch)
        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print(f"The nodes batch: {nodes_batch}")
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        #print(len(nodes_batch))
        visited_nodes |= set(nodes_batch)

        # print(visited_nodes)

        # get ground-truth for the nodes batch
        # labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage(nodes_batch)
        #print(embs_batch)

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
            #print(embs_batch)

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage