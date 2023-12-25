import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp


class Model(nn.Module):
    def __init__(self, sparse_norm_adj, u_graph, i_graph, feat_dim, layer_num, batch_size, device, n_user, n_item, drop_ratio, type_):
        super().__init__()
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.device = device
        self.n_user = n_user
        self.n_item = n_item
        self.drop_ratio = drop_ratio

        self.sparse_norm_adj = self.csr2tensor(sparse_norm_adj)
        self.u_graph = u_graph
        self.i_graph = i_graph
        self.ssl_tau = 0.1

        self.type = type_
        self.user_embedding = torch.nn.Embedding(self.n_user, self.feat_dim)
        self.item_embedding = torch.nn.Embedding(self.n_item, self.feat_dim)
        self._init_model()
        self.f = nn.Sigmoid()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        # initializer = nn.init.xavier_normal_
        initializer(self.user_embedding.weight, gain=1)
        initializer(self.item_embedding.weight, gain=1)

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embedding.weight[users]
        pos_emb_ego = self.item_embedding.weight[pos_items]
        neg_emb_ego = self.item_embedding.weight[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def csr2tensor(self, matrix: sp.csr_matrix):
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)), matrix.shape
        ).to(self.device)
        return x

    def rand_sample(self, high, size=None, replace=True):
        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def create_adjust_matrix(self, sp_adj):

        lenth, _ = sp_adj.get_shape()
        node_idx, node_idy = sp_adj.nonzero()
        if self.type == "ND":
            drop_node = self.rand_sample(lenth, size=int(lenth * self.drop_ratio), replace=False)
            R_node = np.ones(lenth, dtype=np.float32)
            R_node[drop_node] = 0.
            R_node = sp.diags(R_node)
            R_G = sp.csr_matrix((np.ones_like(node_idx, dtype=np.float32), (node_idx, node_idy)),
                                shape=(lenth, lenth))
            res = R_node.dot(R_G)
            res = res.dot(R_node)
            matrix = res

        elif self.type == "ED" or self.type == "RW":
            keep_idx = self.rand_sample(len(node_idx), size=int(len(node_idx) * (1 - self.drop_ratio)), replace=False)
            node_row = node_idx[keep_idx]
            node_col = node_idy[keep_idx]
            matrix = sp.csr_matrix((np.ones_like(node_row), (node_row, node_col)), shape=(lenth, lenth))
        else:
            matrix = sp_adj

        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)


    def graph_construction(self, sp_adj):
        sub_graph1 = []
        if self.type == "ND" or self.type == "ED":
            sub_graph1 = self.csr2tensor(self.create_adjust_matrix(sp_adj))
        elif self.type == "RW":
            for i in range(self.layer_num):
                _g = self.csr2tensor(self.create_adjust_matrix(sp_adj))
                sub_graph1.append(_g)

        sub_graph2 = []
        if self.type == "ND" or self.type == "ED":
            sub_graph2 = self.csr2tensor(self.create_adjust_matrix(sp_adj))
        elif self.type == "RW":
            for i in range(self.layer_num):
                _g = self.csr2tensor(self.create_adjust_matrix(sp_adj))
                sub_graph2.append(_g)
        return sub_graph1, sub_graph2

    def computer(self):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # yelp remove initial representation
        embs = [all_emb]
        g_droped = self.sparse_norm_adj
        for layer in range(self.layer_num):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users_embeddings, items_embeddings = torch.split(light_out, [self.n_user, self.n_item])
        return users_embeddings, items_embeddings

    def get_sub_embed(self, g, h):
        # yelp remove initial representation
        embs = [h]
        if isinstance(g, list):
            for sub_graph in g:
                h = torch.sparse.mm(sub_graph, h)
                embs.append(h)
        else:
            for i in range(self.layer_num):
                h = torch.sparse.mm(g, h)
                embs.append(h)
        embs = torch.stack(embs, dim=1)
        embeddings = torch.mean(embs, dim=1)
        return embeddings

    def InfoNCE(self, view1, view2, temperature=0.1, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def cal_cl_loss(self, user_sub1, user_sub2, item_sub1, item_sub2, batch_users, batch_positive):
        u_emd1 = self.get_sub_embed(user_sub1, self.user_embedding.weight)
        u_emd2 = self.get_sub_embed(user_sub2, self.user_embedding.weight)

        i_emd1 = self.get_sub_embed(item_sub1, self.item_embedding.weight)
        i_emd2 = self.get_sub_embed(item_sub2, self.item_embedding.weight)

        u_idx = torch.unique(batch_users)
        i_idx = torch.unique(batch_positive)

        cl_user_loss = self.InfoNCE(u_emd1[u_idx], u_emd2[u_idx])
        cl_item_loss = self.InfoNCE(i_emd1[i_idx], i_emd2[i_idx])
        return cl_user_loss, cl_item_loss

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss