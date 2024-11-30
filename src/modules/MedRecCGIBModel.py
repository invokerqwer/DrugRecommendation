from .gnn import GNNGraph,GNNGraph_CGIB

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from .gnn import GCN,VQVAE


class MedRecCGIBModel(torch.nn.Module):
    def __init__(
        self, global_para, emb_dim, voc_size,
        ddi_adj,
        weighted_rnn=False, codebook_size=128,
        device=torch.device('cpu'), dropout=0.5,alpha=1, *args, **kwargs
    ):
        super(MedRecCGIBModel, self).__init__(*args, **kwargs)
        self.device = device
        self.voc_size = voc_size
        self.weighted_rnn = weighted_rnn
        self.global_encoder = GNNGraph(**global_para)
        self.cgib = GNNGraph_CGIB(self.device,**global_para)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=1),
            torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, voc_size[2])
        )
        # Instantiate VQ-VAE
        self.vqvae = VQVAE(input_key_dim=emb_dim,input_value_dim=self.voc_size[2], codebook_size=codebook_size, hidden_size=emb_dim,alpha=alpha)

        self.ddi_gcn = GCN(
            voc_size=voc_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.inter.data.uniform_(-initrange, initrange)

    def forward(
        self, mol_data, patient_data,
       tensor_ddi_adj, average_projection,bottleneck = False, 
    ):
        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)
        seq2 = torch.cat(seq2, dim=1)
        seq1 = seq1.permute(1, 0, 2)
        seq2 = seq2.permute(1, 0, 2)
        # print('seq1.shape',seq1.shape)
        # print('seq2.shape',seq2.shape)
        output1 = self.seq_encoders[0](seq1)
        output2 = self.seq_encoders[1](seq2)
        output1 = output1.permute(1, 0, 2) 
        output2 = output2.permute(1, 0, 2)
        # print('output1.shape',output1.shape)
        # print('output2.shape',output2.shape)
        patient_reprs = torch.cat([output1,output2], dim=-1).squeeze(dim=0)
        # print('patient_reprs.shape',patient_reprs.shape)
        queries = self.query(patient_reprs)
        # print('queries.shape',queries.shape)
        query = queries[-1:]  # (1,dim)
        # print('query.shape',query.shape)
        if bottleneck == True:
            global_embeddings,KL_Loss,preserve_rate,patient_pred_loss = self.cgib(query,**mol_data) #(283,dim) (283 is smile type num)
        else :
            global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings) #(131,dim)
        molecule_embeddings = global_embeddings 
        # Initialize vq_loss
        vq_loss = 0.0
        if len(patient_data) > 1 and not bottleneck:
            history_keys, history_values = [], []
            for idx, adm in enumerate(patient_data[:-1]):
                history_keys.append(queries[idx:idx+1].detach())
                history_value = torch.zeros(self.voc_size[2]).to(self.device)
                history_value[adm[2]] = 1
                history_values.append(history_value.unsqueeze(0))
            
            history_keys = torch.cat(history_keys, dim=0)
            history_values = torch.cat(history_values, dim=0)
            
            decoded_keys, decoded_values, encoded_keys, encoded_values, quantized_keys, quantized_values = self.vqvae(history_keys, history_values)
            vq_loss = self.vqvae.compute_loss(history_keys, history_values, decoded_keys, decoded_values, quantized_keys, quantized_values)
        else:
            quantized_keys = self.vqvae.codebook_keys.weight  
            quantized_values = self.vqvae.codebook_values.weight 

        key_weights1 = F.softmax(torch.mm(query, molecule_embeddings.t()), dim=-1) 
        fact1 = torch.mm(key_weights1, molecule_embeddings) 
        if quantized_keys.size(0) > 0:
            visit_weight = F.softmax(torch.mm(query, quantized_keys.t()))
            weighted_values = visit_weight.mm(quantized_values)
            fact2 = torch.mm(weighted_values, molecule_embeddings)
        else:
            fact2 = fact1
            
        output_embedding = torch.cat([query, fact1, fact2], dim=-1)
        score = self.output(output_embedding)
        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        if bottleneck == True:
            return score, batch_neg, KL_Loss, patient_pred_loss, preserve_rate
        else:
            return score, batch_neg,vq_loss  
