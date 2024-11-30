import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from .GNNConv import GNN_node, GNN_node_Virtualnode


from torch_scatter import scatter_mean, scatter_add, scatter_std

class GNNGraph(torch.nn.Module):

    def __init__(
        self, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNGraph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph


class GNNGraph_CGIB(torch.nn.Module):

    def __init__(
        self,device, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNGraph_CGIB, self).__init__()
        self.device = device
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        self.mse_loss = torch.nn.MSELoss()

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        self.compressor = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.BatchNorm1d(self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(self.emb_dim, 1)
                    )
        self.patient_predictor = nn.Linear(self.emb_dim, 1)
        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def compress(self, drug_features):
        p = self.compressor(drug_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p
    
    def forward(self,patient_repr,batched_data):
        drug = batched_data
        drug_features = self.gnn_node(drug) # (7122,dim)
        # patient_repr (1,64)
        lambda_pos, p = self.compress(drug_features)
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos
        
        sim_scores = torch.matmul(drug_features, patient_repr.T)  # 维度 (7122, 1)
        sim_weights = F.softmax(sim_scores, dim=0)  # 维度 (7122, 1)
        drug_features = drug_features * sim_weights  # 广播至 (7122, dim)
        drug_features = F.normalize(drug_features, dim = 1)
        
        # Get Stats
        preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()
        static_drug_feature = drug_features.clone().detach()
        node_feature_mean = scatter_mean(static_drug_feature, drug.batch, dim = 0)[drug.batch]
        node_feature_std = scatter_std(static_drug_feature, drug.batch, dim = 0)[drug.batch]
        
        noisy_node_feature_mean = lambda_pos * drug_features + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        noisy_drug_subgraphs = self.pool(noisy_node_feature, drug.batch) #(283,dim)

        epsilon = 1e-7

        KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), drug.batch).reshape(-1, 1) + \
                    scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), drug.batch, dim = 0)
        KL_Loss = torch.mean(KL_tensor)
        
        patient_pred_loss = self.mse_loss(patient_repr, self.patient_predictor(noisy_drug_subgraphs))
        return noisy_drug_subgraphs,KL_Loss,preserve_rate,patient_pred_loss


class GNN(torch.nn.Module):
    def __init__(
        self, num_tasks, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        super(GNN, self).__init__()
        self.model = GNNGraph(
            num_layer, emb_dim, gnn_type,
            virtual_node, residual, drop_ratio, JK, graph_pooling
        )
        self.num_tasks = num_tasks
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(
                2 * self.model.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.model.emb_dim, self.num_tasks
            )

    def forward(self, batched_data):
        h_graph = self.model(batched_data)
        return self.graph_pred_linear(h_graph)



import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import numpy as np
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # adj = self.normalize(adj + np.eye(adj.shape[0]))
        adj = self.normalize((adj + torch.eye(adj.shape[0], device=adj.device)).cpu().numpy())

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


if __name__ == '__main__':
    GNN(num_tasks=10)
