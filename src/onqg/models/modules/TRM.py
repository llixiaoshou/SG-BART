

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)

    # def forward(self, Q, K, V,attn_mask, pos_edges_hidden, spanmask_deges_hidden):
    def forward(self, Q, K, V, enc_inputs_mask,pos_edges_hidden, spanmask_deges_hidden):
        ## 输入进来的维度分别是 Q: [batch_size x n_heads x len_q x d_q]  K： [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]

        #print('Q',Q.shape)

        # print('pos_edges_hidden',pos_edges_hidden.shape)
        # ori_dim = K.size(-1)
        # #pos_edges_hidden =  pos_edges_hidden.view(K.size(0), K.size(1), K.size(2), -1)
        #
        edgesi=pos_edges_hidden.view(Q.size(0), Q.size(2),Q.size(2),-1,8).permute(0,4,1,2,3)
        # # B,L,L,512   B,L,L,64,8  B,8,L,L,64
        # # print('3::',edgesi.shape)
        # #K_e = Ki+edgesi
        # #V_e = Vi+edgesi
        # # print('####',Q.unsqueeze(-1).shape)
        # # print('$$$$',K.unsqueeze(-3).transpose(-2,-1).shape)
        # # print('%%%%',edgesi.shape)
        scoresi = (Q.unsqueeze(-1)*K.unsqueeze(-3).transpose(-2,-1)+Q.unsqueeze(-1)*edgesi.transpose(-2,-1)).sum(-2) / np.sqrt(64)
        #
        #
        # '''Q = Q.reshape(-1, Q.size(2), Q.size(3))
        # print('Q', Q.shape)
        # K = K.reshape(-1, K.size(2), K.size(3))
        # scoresi = torch.bmm(Q.unsqueeze(-1), K.unsqueeze(-3).transpose(-2,-1)) + torch.bmm(Q.unsqueeze(-1),edgesi.transpose(-2,-1))/np.sqrt(64)
        # print('scoresi',scoresi.shape)'''
        # # scoresi = scoresi+M.unsqueeze(1).repeat(1,8,1,1)
        if enc_inputs_mask is not None:
            pad_attn_mask = enc_inputs_mask.unsqueeze(1)
            # print('pad_attn_mask',pad_attn_mask.shape)
            pad_attn_mask = pad_attn_mask.expand(Q.size(0), Q.size(2), K.size(2))
            attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, 8, 1, 1).cuda()
            attn_mask = torch.tensor(attn_mask, dtype=bool).cuda()
            scoresi = scoresi.masked_fill_(attn_mask, torch.tensor(-1e9))

        attn = nn.Softmax(dim=-1)(scoresi)
        attn = self.dropout(attn)

        context = (attn.unsqueeze(-1)*V.unsqueeze(-3)+attn.unsqueeze(-1)*edgesi).sum(-2)


        # print('output::',context.shape)
        # assert 0==1
        # Q = torch.cat([Q, pos_edges_hidden.float()], dim=-1).cuda()
        # dim1 = Q .size(-1)
        # dim2 = ori_dim
        # Q = nn.Linear(dim1, dim2).cuda()(Q)
        #
        # K = torch.cat([K, pos_edges_hidden.float()], dim=-1).cuda()
        # dim1 = K.size(-1)
        # dim2 = ori_dim
        # K = nn.Linear(dim1, dim2).cuda()(K)
        #
        # # print("K+",K.shape)
        #
        # # K = nn.Linear(dim1, dim2).cuda()(K)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64)
        if enc_inputs_mask is not None:
            pad_attn_mask = enc_inputs_mask.unsqueeze(1)

            pad_attn_mask = enc_inputs_mask.unsqueeze(1)
            # print('pad_attn_mask',pad_attn_mask.shape)
            pad_attn_mask=pad_attn_mask.expand(Q.size(0), Q.size(2), K.size(2))
            attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, 8, 1, 1).cuda()
            attn_mask = torch.tensor(attn_mask, dtype=bool).cuda()
            scores=scores.masked_fill_(attn_mask, torch.tensor(-1e9))

        attn = nn.Softmax(dim=-1)(scores)
        attn=self.dropout(attn)

        context = torch.matmul(attn, V).cuda()

        #
        # # print('scores.shape+++', scores.shape)
        # # Q(K + De)+M
        # # scores.masked_fill_(attn_mask, -1e9)
        # # print('spanmask_deges_hidden+.shape', spanmask_deges_hidden.shape)
        # # spanmask_deges_mask= spanmask_deges_hidden.unsqueeze(1)
        # #
        # #
        # #
        # # scores = torch.matmul(scores, spanmask_deges_mask.float())
        #
        #
        #
        # # print('scores',  scores.shape)
        #
        # # scores = torch.cat([scores.to(torch.float), spanmask_deges_hidden.to(torch.float)], dim=-1)
        # # V+De
        #
        # # print('V',V.shape)
        # ori_dim = V.size(-1)
        #
        # pos_edges_hidden = pos_edges_hidden.view(V.size(0), V.size(1), V.size(2), -1)
        #
        #
        # V = torch.cat([V.to(torch.float), pos_edges_hidden.to(torch.float)], dim=-1)
        #
        # dim1 = V.size(-1)
        # dim2 = ori_dim
        # V = nn.Linear(dim1, dim2).cuda()(V)
        #
        #
        # # Softmax(Q(K + De)+M)
        # dim1 = scores.size(-1)
        # dim2 = V.size(2)
        #
        #
        # if enc_inputs_mask is not None:
        #     pad_attn_mask = enc_inputs_mask.unsqueeze(1)
        #
        #     pad_attn_mask = enc_inputs_mask.unsqueeze(1)
        #     # print('pad_attn_mask',pad_attn_mask.shape)
        #     pad_attn_mask=pad_attn_mask.expand(Q.size(0), Q.size(2), K.size(2))
        #     attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, 8, 1, 1).cuda()
        #     attn_mask = torch.tensor(attn_mask, dtype=bool).cuda()
        #     scores=scores.masked_fill_(attn_mask, torch.tensor(-1e9))
        #
        # attn = nn.Softmax(dim=-1)(scores)
        # attn=self.dropout(attn)
        #
        # context = torch.matmul(attn, V).cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # print('context',context.shape)

        return context, attn


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(512, 64 * 8)
        self.W_K = nn.Linear(512, 64 * 8)
        self.W_V = nn.Linear(512, 64 * 8)
        self.linear = nn.Linear(8 * 64, 512)
        self.layer_norm = nn.LayerNorm(512)

    # def forward(self, Q, K, V, attn_mask,pos_edges_hidden, spanmask_deges_hidden):
    def forward(self, Q, K, V, enc_inputs_mask,  pos_edges_hidden, spanmask_deges_hidden):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, 8, 64).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, 8, 64).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, 8, 64).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, 8, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, enc_inputs_mask,  pos_edges_hidden, spanmask_deges_hidden)

        # context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask ,pos_edges_hidden, spanmask_deges_hidden)

        # print('context.shape',context.shape)
        #
        # print('++++', context.transpose(1, 2).shape)


        context = context.transpose(1, 2).contiguous().view(batch_size, -1, 8 * 64) # context: [batch_size x len_q x n_heads * d_v]


        output = self.linear(context)
        # print('output',output.shape)
        # print('residual',residual.shape)

        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


def get_attn_pad_mask(seq_q, seq_k):
    # print('++++++',seq_q.size())
    batch_size, len_q,m = seq_q.size()
    batch_size, len_k,n = seq_k.size()
    # eq(zero) is PAD token
    seq_k=torch.sum(seq_k,2)

    # print('seq_k',seq_k.shape)

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # print(' pad_attn_mask ', pad_attn_mask .shape)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    # def forward(self, enc_inputs,enc_self_attn_mask, pos_edges_hidden, spanmask_deges_hidden):
    def forward(self, enc_inputs,enc_inputs_mask,pos_edges_hidden, spanmask_deges_hidden):

        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_inputs_mask, pos_edges_hidden, spanmask_deges_hidden) # enc_inputs to same Q,K,V
        # enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask, pos_edges_hidden, spanmask_deges_hidden)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


## 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.pos_emb = PositionalEncoding(d_model=512)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(6)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs, enc_inputs_mask,pos_edges_hidden, spanmask_deges_hidden):

    # def forward(self, enc_inputs,  edge_in_hidden, spanmask_deges_hidden):
    #     enc_self_attns = []

        enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)

        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)


        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_inputs,enc_inputs_mask, pos_edges_hidden, spanmask_deges_hidden)
            # enc_outputs, enc_self_attn = layer(enc_inputs, enc_self_attn_mask,  pos_edges_hidden, spanmask_deges_hidden)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


## 1. 从整体网路结构来看，分为三个部分：编码层
class Transformer(nn.Module):
    def __init__(self,device,droput=0.5):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层
        self.device=device
        self.droput=droput
        # self.output_gate = Propagator(d_model, dropout=dropout)
        self.gate = nn.Linear(512* 2, 512, bias=False)

    def gated_output(self, outputs, inputs):
        concatenation = torch.cat((outputs, inputs), dim=2)
        g_t = torch.sigmoid(self.gate(concatenation))

        output = g_t * outputs + (1 - g_t) * inputs
        return output

    def forward(self,nodes, enc_inputs,enc_inputs_mask, pos_edges_hidden, spanmask_deges_hidden):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs.cuda(), enc_inputs_mask.cuda(), pos_edges_hidden.cuda(), spanmask_deges_hidden.cuda())

        enc_outputs= self.gated_output( enc_outputs, nodes)

        return enc_outputs





