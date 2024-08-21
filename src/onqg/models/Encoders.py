import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import GatedSelfAttention, ConcatAttention
from onqg.models.modules.Layers import GraphEncoderLayer, SparseGraphEncoderLayer
from onqg.models.modules.TRM import Transformer
# from pytorch_pretrained_bert import BertModel
import numpy as np


class RNNEncoder(nn.Module):
    """
    Input: (1) inputs['src_seq']
           (2) inputs['lengths'] 
           (3) inputs['feat_seqs']
    Output: (1) enc_output
            (2) hidden
    """
    def __init__(self, n_vocab, d_word_vec, d_model, n_layer,brnn, rnn, feat_vocab, d_feat_vec, slf_attn,  dropout,ans_n_vocab,d_k=64):
        self.name = 'rnn'
        self.n_layer = n_layer
        self.num_directions = 2 if brnn else 1
        assert d_model % self.num_directions == 0, "d_model = hidden_size x direction_num"
        self.hidden_size = d_model // self.num_directions
        self.d_enc_model = d_model
        self.d_k = d_k

        super(RNNEncoder, self).__init__()

        self.attn = ConcatAttention(d_model, d_model, d_k)


        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.ans_emb = nn.Embedding(ans_n_vocab, d_word_vec, padding_idx=Constants.PAD)
        input_size = d_word_vec

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            input_size += len(feat_vocab) * d_feat_vec

        self.slf_attn = slf_attn
        if slf_attn:
            self.gated_slf_attn = GatedSelfAttention(d_model)

        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=n_layer,
                               dropout=dropout, bidirectional=brnn, batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=n_layer,
                              dropout=dropout, bidirectional=brnn, batch_first=True)
        else:
            raise ValueError("Only support 'LSTM' and 'GRU' for RNN-based Encoder ")

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['brnn'], opt['rnn'], opt['feat_vocab'], opt['d_feat_vec'],
                   opt['slf_attn'], opt['dropout'],opt['ans_n_vocab'])

    def forward(self, inputs):
        # print('*******************************************')
        # print(inputs)


        src_seq, lengths, feat_seqs = inputs['src_seq'], inputs['lengths'], inputs['feat_seqs']

        lengths = torch.LongTensor(lengths.data.view(-1).tolist())

        enc_input = self.word_emb(src_seq)

        if self.feature:
            feat_outputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_outputs = torch.cat(feat_outputs, dim=2)
            enc_input = torch.cat((enc_input, feat_outputs), dim=-1)

        enc_input = pack(enc_input, lengths, batch_first=True, enforce_sorted=False)

        enc_output, hidden = self.rnn(enc_input, None)
        enc_output = unpack(enc_output, batch_first=True)[0]

        if self.slf_attn:
            mask = (src_seq == Constants.PAD).byte()
            enc_output = self.gated_slf_attn(enc_output, mask)
            # print('mask',mask)
        # print(' enc_output', enc_output)
        # print('enc_output', enc_output.size())

        # answer
        ans_seq, ans_lengths = inputs['ans_seq'], inputs['ans_lengths']
        ans_mask=inputs['ans_mask']
        # print('ans_seq',ans_seq)
        ans_lengths = torch.LongTensor(ans_lengths.data.view(-1).tolist())
        ans_input = self.ans_emb(ans_seq)
        ans_input = pack(ans_input, ans_lengths, batch_first=True, enforce_sorted=False)
        ans_output, ans_hidden = self.rnn(ans_input, None)
        ans_output = unpack(ans_output, batch_first=True)[0]

        ans_output=self.avg_pool( ans_output,ans_mask)
        if self.slf_attn:
            ans_mask = (ans_seq == Constants.PAD).byte()
            ans_output = self.gated_slf_attn(ans_output, ans_mask)
        #
        # ans_words = torch.sum(self.ans_emb(ans_seq), dim=1)
        return enc_output, hidden, ans_output, ans_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2).repeat(1,1, hidden_states.size(-1)).cuda()
        # print(mask.shape)
        hidden = hidden_states.masked_fill(mask == 0, 0.0).cuda()
        # print(hidden)
        avg_hidden = torch.sum(hidden, 1) / length.cuda()
        return avg_hidden

class GraphEncoder(nn.Module):
    """Combine GGNN (Gated Graph Neural Network) and GAT (Graph Attention Network)
    Input: (1) nodes - [batch_size, node_num, d_model]
           (2) edges - ([batch_size, node_num * node_num], [batch_size, node_num * node_num]) 1st-inlink, 2nd-outlink
           (3) mask - ([batch_size, node_num, node_num], [batch_size, node_num, node_num]) 1st-inlink, 2nd-outlink
           (4) node_feats - list of [batch_size, node_num]
    """

    def __init__(self, n_edge_type, d_model, n_layer, alpha, d_feat_vec,
                 feat_vocab, layer_attn, dropout, attn_dropout):
        self.name = 'graph'
        super(GraphEncoder, self).__init__()
        self.layer_attn = layer_attn

        self.hidden_size = d_model
        self.d_model = d_model
        ###=== node features ===###
        self.feature = True if feat_vocab else False
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            # self.hidden_size += d_feat_vec * len(feat_vocab)
            self.feature_transform = nn.Linear(self.hidden_size + d_feat_vec * len(feat_vocab), self.hidden_size)
        ###=== edge embedding ===###
        # self.edge_in_emb = nn.Embedding(n_edge_type, self.hidden_size * d_model, padding_idx=Constants.PAD)
        # self.edge_out_emb = nn.Embedding(n_edge_type, self.hidden_size * d_model, padding_idx=Constants.PAD)
        # self.edge_bias = edge_bias
        # if edge_bias:
        #     self.edge_in_emb_bias = nn.Embedding(n_edge_type, d_model, padding_idx=Constants.PAD)
        #     self.edge_out_emb_bias = nn.Embedding(n_edge_type, d_model, padding_idx=Constants.PAD)
        ###=== graph encode layers===###
        self.layer_stack = nn.ModuleList([
            GraphEncoderLayer(self.hidden_size, d_model, alpha, feature=self.feature,
                              dropout=dropout, attn_dropout=attn_dropout) for _ in range(n_layer)
        ])
        ###=== gated output ===###
        self.gate = nn.Linear(2 * d_model, d_model, bias=False)

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_edge_type'], opt['d_model'], opt['n_layer'], opt['alpha'],
                   opt['d_feat_vec'], opt['feat_vocab'], opt['layer_attn'],
                   opt['dropout'], opt['attn_dropout'])

    def gated_output(self, outputs, inputs):
        concatenation = torch.cat((outputs, inputs), dim=2)
        g_t = torch.sigmoid(self.gate(concatenation))

        output = g_t * outputs + (1 - g_t) * inputs
        return output

    def forward(self, inputs):

        # print('++++++++++')

        nodes, mask = inputs['nodes'], inputs['mask']
        node_feats, node_type = inputs['feat_seqs'], inputs['type']
        nodes = self.activate(nodes)
        node_output = nodes  # batch_size x node_num x d_model
        ###=== get embeddings ===###
        feat_hidden = None
        if self.feature:
            feat_hidden = [feat_emb(node_feat) for node_feat, feat_emb in zip(node_feats, self.feat_embs)]
            feat_hidden = torch.cat(feat_hidden, dim=2)  # batch_size x node_num x (hidden_size - d_model)
            node_output = self.feature_transform(torch.cat((node_output, feat_hidden), dim=-1))
        # batch_size x (node_num * node_num) x hidden_size x d_model
        # edge_in_hidden = self.edge_in_emb(edges[0]).view(nodes.size(0), -1, self.hidden_size, nodes.size(2))
        # edge_out_hidden = self.edge_out_emb(edges[1]).view(nodes.size(0), -1, self.hidden_size, nodes.size(2))
        # edge_hidden = (edge_in_hidden, edge_out_hidden)
        # if self.edge_bias:
        #     # batch_size x (node_num * node_num) x d_model
        #     edge_in_hidden_bias, edge_out_hidden_bias = self.edge_in_emb_bias(edges[0]), self.edge_out_emb_bias(edges[1])
        # edge_hidden_bias = (edge_in_hidden_bias, edge_out_hidden_bias) if self.edge_bias else None
        ##=== forward ===###
        node_outputs = []
        for enc_layer in self.layer_stack:
            # node_output = enc_layer(node_output, edge_hidden, mask, feat_hidden=feat_hidden,
            #                         edge_hidden_bias=edge_hidden_bias)
            node_output = enc_layer(node_output, mask, node_type, feat_hidden=feat_hidden)
            node_outputs.append(node_output)

        node_output = self.gated_output(node_output, nodes)
        node_outputs[-1] = node_output
        hidden = [layer_output.transpose(0, 1)[0] for layer_output in node_outputs]
        if self.layer_attn:
            node_output = node_outputs
        return node_output, hidden


class SparseGraphEncoder(nn.Module):
    """Sparse version of Graph Encoder"""
    """Combine GGNN (Gated Graph Neural Network) and GAT (Graph Attention Network)
    Input: (1) nodes - [batch_size, node_num, d_model]
           (2) edges - [edge_type_num]
           (3) mask - ([batch_size, node_num, node_num], [batch_size, node_num, node_num]) 1st-inlink, 2nd-outlink
           (4) node_feats - list of [batch_size, node_num]
           (5) adjacent_matrix - 2 * [batch_size, real_node_num, real_neighbor_num] 1st-inlink, 2nd-outlink
    """

    def __init__(self, n_edge_type, d_model, d_rnn_enc_model, n_layer, alpha, d_feat_vec,
                 feat_vocab, layer_attn, dropout, attn_dropout):
        self.name = 'graph'
        super(SparseGraphEncoder, self).__init__()
        self.layer_attn = layer_attn

        self.hidden_size = d_model
        self.d_model = d_model
        ###=== node features ===###
        self.feature = True if feat_vocab else False
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            self.hidden_size += d_feat_vec * len(feat_vocab)
            ###=== edge embedding ===###
        self.edge_in_emb = nn.Embedding(n_edge_type, self.hidden_size * d_model, padding_idx=Constants.PAD)
        self.edge_out_emb = nn.Embedding(n_edge_type, self.hidden_size * d_model, padding_idx=Constants.PAD)
        self.edge_bias = True
        if self.edge_bias:
            self.edge_in_emb_bias = nn.Embedding(n_edge_type, d_model, padding_idx=Constants.PAD)
            self.edge_out_emb_bias = nn.Embedding(n_edge_type, d_model, padding_idx=Constants.PAD)
        ###=== graph encode layers===###
        self.layer_stack = nn.ModuleList([
            SparseGraphEncoderLayer(self.hidden_size, d_model, alpha, edge_bias=self.edge_bias, feature=self.feature,
                                    dropout=dropout, attn_dropout=attn_dropout) for _ in range(n_layer)
        ])
        ###=== gated output ===###
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_edge_type'], opt['d_model'], opt['d_rnn_enc_model'], opt['n_layer'], opt['alpha'],
                   opt['d_feat_vec'], opt['feat_vocab'], opt['edge_bias'], opt['layer_attn'],
                   opt['dropout'], opt['attn_dropout'])

    def gated_output(self, outputs, inputs):
        concatenation = torch.cat((outputs, inputs), dim=2)
        g_t = torch.sigmoid(self.gate(concatenation))

        output = g_t * outputs + (1 - g_t) * inputs
        return output

    def forward(self, inputs):
        nodes, edges, mask = inputs['nodes'], inputs['edges'], inputs['mask']
        node_feats, adjacent_matrix = inputs['feat_seqs'], inputs['adjacent_matrix']
        nodes = self.activate(nodes)
        node_output = nodes  # batch_size x node_num x d_model
        ###=== get embeddings ===###
        feat_hidden = None
        if self.feature:
            feat_hidden = [feat_emb(node_feat) for node_feat, feat_emb in zip(node_feats, self.feat_embs)]
            feat_hidden = torch.cat(feat_hidden, dim=2)  # batch_size x node_num x (hidden_size - d_model)
        # batch_size x (node_num * node_num) x hidden_size x d_model
        edge_in_hidden = self.edge_in_emb(edges).view(-1, self.hidden_size, nodes.size(2))
        edge_out_hidden = self.edge_out_emb(edges).view(-1, self.hidden_size, nodes.size(2))
        edge_hidden = (edge_in_hidden, edge_out_hidden)
        if self.edge_bias:
            # batch_size x (node_num * node_num) x d_model
            edge_in_hidden_bias, edge_out_hidden_bias = self.edge_in_emb_bias(edges), self.edge_out_emb_bias(edges)
        edge_hidden_bias = (edge_in_hidden_bias, edge_out_hidden_bias) if self.edge_bias else None
        ###=== forward ===###
        node_outputs = []
        for enc_layer in self.layer_stack:
            node_output = enc_layer(node_output, edge_hidden, mask, adjacent_matrix,
                                    feat_hidden=feat_hidden, edge_hidden_bias=edge_hidden_bias)
            node_outputs.append(node_output)

        node_output = self.gated_output(node_output, nodes)
        node_outputs[-1] = node_output

        hidden = [layer_output.transpose(0, 1)[0] for layer_output in node_outputs]

        if self.layer_attn:
            node_output = node_outputs

        return node_output, hidden


class EncoderTransformer(nn.Module):
    """Transform RNN-Encoder's output to Graph-Encoder's input
    Input: seq_output - [batch_size, seq_length, rnn_enc_dim] (tensor)
           root_list - [batch_size, node_num] (list)
           indexes_list - [batch_size, node_num, index_num] (list)
           node_sizes - [batch_size * node_num, 1] (list)
    """
    def __init__(self, d_model,  d_feat_vec,feat_vocab, d_k=64,  device=None):
        super(EncoderTransformer, self).__init__()
        self.device = device
        self.d_k = d_k
        self.hidden_size1 = d_model
        # attend_dim, query_dim, att_dim
        self.attn = ConcatAttention(d_model, d_model, d_k)
        # self.trans_encoder = Transformer(d_model,d_k)

        self.feature = True if feat_vocab else False
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, 256, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            # self.hidden_size += d_feat_vec * len(feat_vocab)
            self.feature_transform = nn.Linear(self.hidden_size1*2, self.hidden_size1)



    def forward(self, inputs, max_length):


        # print('inputs',inputs)

        def pad(vectors, data_length, max_length=None):
            hidden_size = (max_length, vectors.size(1))
            out = torch.zeros(hidden_size, device=self.device)
            out.narrow(0, 0, data_length).copy_(vectors)    # bag_size x rnn_enc_dim
            return out

        seq_output, hidden, indexes_list = inputs['seq_output'], inputs['hidden'], inputs['index']
        node_feats, node_type = inputs['feat_seqs'], inputs['type']


        # print('seq_output', seq_output.shape)
        # print('indexes_list', len(indexes_list))
        # print('hidden', hidden.shape)
        # print(indexes_list)


        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)


        hidden = hidden.contiguous().view(hidden.size(0), -1)

        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # print('hidden',hidden.shape)
        # print('hidden_shape', hidden.shape)
        # print('hidden_shape', hidden.shape)

        # root_list = inputs['root']
        node_sizes, node_lengths = inputs['lengths'], inputs['node_lengths']
        # print('node_sizes', node_sizes)
        # print('node_lengths', node_lengths)
        max_length = max(node_sizes)
        ##===== prepare vectors (do padding) =====##
        roots, bags, cnt = [], [], 0
        node_indexs = []
        # for sample_idx, sample in enumerate(zip(root_list, indexes_list)):
        for sample_idx, indexes in enumerate(indexes_list):
            # root, indexes = sample[0], sample[1]
            # for root_idx, indexes_idx in zip(root, indexes):
            for indexes_idx in indexes:
                # roots.append(seq_output[sample_idx][root_idx])
                roots.append(hidden[sample_idx])

                bag = pad(torch.stack([seq_output[sample_idx][idx] for idx in indexes_idx], dim=0),
                          node_sizes[cnt], max_length)
                bags.append(bag)
                cnt += 1
        roots = torch.stack(roots, dim=0)   # all_node_num x rnn_enc_dim
        # print('*****************',roots.shape)


        # print('roots_shape', roots.shape)

        bags = torch.stack(bags, dim=0)     # all_node_num x bag_size x rnn_enc_dim

        # print('**********************',bags.shape)

        ##===== cross attention =====##
        context, *_ = self.attn(roots, bags)    # all_node_num x rnn_enc_dim


        ##===== get node vectors =====##
        max_length = max(node_lengths)
        nodes = []
        nodes_mask = []
        for node_length in node_lengths:
            nodes.append(pad(context[:node_length], node_length, max_length))
            node_mask = torch.zeros(max_length)
            node_mask[:node_length] = 1
            nodes_mask.append(node_mask)
            context = context[node_length:]

        nodes_mask = torch.stack(nodes_mask)
        # print('nodes_mask', nodes_mask)
        # print('nodes_mask', nodes_mask.size())
        nodes = torch.stack(nodes, dim=0)   # batch_size x node_num x d_model
        # print('nodes_shape_1', nodes.shape)
        # nodes = self.trans_encoder(nodes)
        # print('nodes_shape_2', nodes.shape())

        # nodes = self.activate(nodes)
        # node_output = nodes    # batch_size x node_num x d_model
        #
        # ###=== get embeddings ===###
        # feat_hidden = None
        # if self.feature:
        #     #获取其向量表示
        #     feat_hidden = [feat_emb(node_feat) for node_feat, feat_emb in zip(node_feats, self.feat_embs)]
        #     # print("feat_hidden",feat_hidden)
        #     feat_hidden = torch.cat(feat_hidden, dim=2)     # batch_size x node_num x (hidden_size - d_model)
        #     node_output = self.feature_transform(torch.cat((node_output, feat_hidden), dim=-1))



        return nodes, nodes_mask, hidden





class TransfEncoder(nn.Module):
    """
    Input: (1) inputs['src_seq']
           (2) inputs['src_pos']
           (3) inputs['feat_seqs']
    Output: (1) enc_output
            (2) hidden
    """
    def __init__(self, n_vocab, pretrained=None, model_name='default', layer_attn=False):
        self.name = 'transf'
        self.model_type = model_name

        super(TransfEncoder, self).__init__()

        self.layer_attn = layer_attn
        self.pretrained = pretrained

        if model_name == 'bert':
            self.d_enc_model = 768
            self.d_head = 8
            self.n_enc_layer = 12
    
    @classmethod
    def from_opt(cls, opt):
        if opt['pretrained'].count('bert'):
            pretrained = BertModel.from_pretrained(opt['pretrained'])
            return cls(opt['n_vocab'], pretrained=pretrained, layer_attn=opt['layer_attn'], model_name='bert')
        else:
            raise ValueError("Other pretrained models haven't been supported yet")

    def forward(self, inputs, return_attns=False):
        src_seq = inputs['src_seq']
        
        if self.model_type == 'bert':
            enc_outputs, *_ = self.pretrained(src_seq, output_all_encoded_layers=True)
            enc_output = enc_outputs[-1]
        
        hidden = [layer_output.transpose(0, 1)[0] for layer_output in enc_outputs]
        if self.layer_attn:
            enc_output = enc_outputs
            
        return enc_output, hidden