import torch
import torch.nn as nn
from torch.autograd import Variable

import onqg.dataset.Constants as Constants
from onqg.models.modules.Attention import ConcatAttention
from onqg.models.modules.TRM import Transformer

from onqg.models.modules.Attention import ConcatAttention


class UnifiedModel(nn.Module):
    ''' Unify Sequence-Encoder and Graph-Encoder

    Input:  seq-encoder: src_seq, lengths, feat_seqs
            graph-encoder: edges

            encoder-transform: index, lengths, root
            decoder: tgt_seq, src_seq, feat_seqs
            answer-encoder: src_seq, lengths, feat_seqs

    Output: results output from the Decoder (type: dict)
    '''

    def __init__(self, model_type, seq_encoder, graph_encoder, encoder_transformer, Ai_transformer,
                 decoder, decoder_transformer, Concat_Attention,adv=False):
        # print('##################################************************************')
        super(UnifiedModel, self).__init__()
        # self.d_model=d_model
        # self.d_k = d_k

        self.attn = Concat_Attention
        self.model_type = model_type

        self.seq_encoder = seq_encoder

        self.encoder_transformer = encoder_transformer
        self.graph_encoder = graph_encoder
        ###增加transformer
        self.transformer_encoder = Ai_transformer

        self.decoder_transformer = decoder_transformer
        self.decoder = decoder
        self.adv=adv

        # gate
        self.W1 = nn.Linear(768, 512)
        self.W2 = nn.Linear(1536, 512)
        # self.W3 = nn.Linear(512, 1024)




    def forward(self, inputs, max_length=None):
        seq_output, hidden, ans_output, ans_hidden = self.seq_encoder(inputs['seq-encoder'])
        # print('ans_output',ans_output.shape)

        ans_len = ans_output.size(1)
        ans_output1=ans_output
        # ans_output= torch.sum(ans_output, dim=1)


        # seq_output_length =  seq_output.size(1)
        # seq_output1 = torch.sum( seq_output, dim=1)
        #
        # seq_output1, *_ = self.attn(seq_output1, ans_output1)
        # seq_output1= seq_output1.unsqueeze(1)
        # seq_output = seq_output1.repeat(1, seq_output_length, 1)
        # # print(' seq_output', seq_output.shape)
        #
        # ans_output = torch.sum(ans_output, dim=1)

        ## encoder transform ##
        inputs['encoder-transform']['seq_output'] = seq_output
        inputs['encoder-transform']['hidden'] = hidden
        node_input,nodes_mask, hidden = self.encoder_transformer(inputs['encoder-transform'], max_length)


        # print('+++++++++_____________________')
        # print('nodes', nodes.shape)

        ## graph encode ##
        inputs['graph-encoder']['nodes'] = node_input
        # inputs['graph-encoder']['nodes_mask'] = nodes_mask

        node_output, edge_in_hidden = self.graph_encoder(inputs['graph-encoder'])

        # pos_edges_hidden, spanmask_deges_hidden = self.graph_encoder(
        #     inputs['graph-encoder'])

        # print(' pos_edges_hidden', pos_edges_hidden.shape)



        # node_output, edge_in_hidden, spanmask_deges_hidden = self.graph_encoder(
        #     inputs['graph-encoder'])

        # node_output, edge_in_hidden = self.graph_encoder(inputs['graph-encoder'])

        # print('node_output',node_output.shape)

        # enc_outputs = self.transformer_encoder(nodes, node_input, nodes_mask, pos_edges_hidden, spanmask_deges_hidden)

        # enc_outputs = self.transformer_encoder(nodes, node_input,nodes_mask,pos_edges_hidden, spanmask_deges_hidden)

        # node_output, edge_in_hidden, spanmask_deges_hidden = self.graph_encoder(inputs['graph-encoder'])

        # enc_outputs = self.transformer_encoder(node_input, edge_in_hidden)

        # print('###################******************************************************')
        # print('seq_output',seq_output.shape)
        # print('enc_outputs',enc_outputs.shape)
        #
        # concatnodes = torch.cat([seq_output, enc_outputs], dim=2)
        # print('********************************#############################', concatnodes.shape)
        # enc_outputs_length = enc_outputs.size(1)
        # enc_outputs1= torch.sum(enc_outputs, dim=1)
        # # enc_outputs_length=enc_outputs
        # enc_outputs1, *_ = self.attn(enc_outputs1,ans_output1)
        # enc_outputs1 = enc_outputs1.unsqueeze(1)
        # enc_outputs= enc_outputs1.repeat(1, enc_outputs_length, 1)

        # print('@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%',context1)

        # new_anss = torch.cat([context, context1], dim=-1)

        # gates
        # new_anss=self.W1(new_anss)

        # print(' new_anss', new_anss.shape)
        # print('ans_output',ans_output1.shape)

        # concats = torch.cat((context,context1, ans_output1), -1)
        # g_t=torch.sigmoid(self.W2(concats))
        # new_ans= g_t * context + (1 - g_t) * context1
        # # #
        # orisize=  new_ans.size(-1)
        # #
        # ans_node_src = nn.Linear(orisize, ans_output.size(-1)).cuda()(new_ans)

        # print('ans_node_src ', ans_node_src.shape)
        # print('ans_node_src ', ans_node_src.shape)

        outputs = {}
        outputs['enc_outputs'] = node_output

        # print('+++++++++++++++')
        # print('enc_outputs', node_output.shape)
        # print('enc_outputs', enc_outputs.shape)

        # #========== classify =========#
        if self.model_type != 'generate':
            scores = self.classifier(node_output) if not self.decoder.layer_attn else self.classifier(node_output[-1])
            inputs['decoder-transform']['scores'] = scores
            outputs['classification'] = scores



            # print('enc_outputs',enc_outputs)
            # print('enc_outputs',enc_outputs.shape)


            # scores = self.classifier(enc_outputs) if not self.decoder.layer_attn else self.classifier(enc_outputs[-1])
            # print('###########################',scores)

            # inputs['decoder-transform']['scores'] = scores

            # print('scores',scores)

            # outputs['classification'] = scores

        # #========== generate =========#
        # print('enc_outputs',enc_outputs.shape)
        # scores = self.classifier(enc_outputs) if not self.decoder.layer_attn else self.classifier(enc_outputs[-1])
        # print('scores',scores.shape)
        # inputs['decoder-transform']['scores'] = scores
        inputs['decoder-transform']['transformer_output'] =node_output
        inputs['decoder-transform']['seq_output'] = seq_output
        inputs['decoder-transform']['hidden'] = hidden

        # print('###################******************************************************')
        # print('seq_output', seq_output.shape)
        # print('enc_outputs', enc_outputs.shape)

        inputs['decoder']['concatnodes1'], inputs['decoder']['scores'], inputs['decoder'][
            'hidden'] = self.decoder_transformer(inputs['decoder-transform'])
        m=inputs['decoder']['concatnodes1']
        # print('m',m.shape)
        m=self.W1(m)
        inputs['decoder']['ans_node_src'] =  ans_output
        # inputs['decoder']['concatnodes']=concatnodes
        inputs['decoder']['ans_len'] = ans_len
        inputs['decoder']['hidden'] = hidden
        # print(' hidden', hidden.shape)

        # print("测试",inputs['decoder'])

        dec_output = self.decoder(inputs['decoder'])
        outputs['concatnodes'] = m
        outputs['generation'] = dec_output


        # print("outputs['generation']['pred']", outputs['generation']['pred'].shape)
        # ========== generate =========#
        if self.model_type != 'classify':
            outputs['generation']['pred'] = self.generator(dec_output['pred'])
            # print("outputs['generation']['pred']", outputs['generation']['pred'].shape)

        outputs['nodes_mask'] = nodes_mask

        # print('outputs',outputs)

        return outputs
