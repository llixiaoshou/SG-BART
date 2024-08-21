import torch
import onqg.dataset.Constants as Constants
from onqg.utils.mask import get_edge_mask, get_edges
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-base')



def preprocess_batch(batch, n_edge_type, sparse=True, feature=False, dec_feature=0, 
                     answer=False, ans_feature=False, copy=False, node_feature=False, 
                     device=None):
    """Get a batch by indexing to the Dataset object, then preprocess it to get inputs for the model
    Input: batch
        raw-index: idxBatch
        src: (wrap(srcBatch), lengths)
        tgt: wrap(tgtBatch)
        copy: (wrap(copySwitchBatch), wrap(copyTgtBatch))
        feat: (tuple(wrap(x) for x in featBatches), lengths)
        ans: (wrap(ansBatch), ansLengths)
        ans_feat: (tuple(wrap(x) for x in ansFeatBatches), ansLengths)

        edges: (wrap(edgeInBatch), wrap(edgeOutBatch), edge_lengths)
        edges_dict: (edgeInDict, edgeOutDict)
        tmp_nodes: (wrap(tmpNodesBatch), node_lengths)
        graph_index: (graphIndexBatch, node_index_length)
        graph_root: graphRootBatch
        node_feat: (tuple(wrap(x) for x in nodeFeatBatches), node_lengths)
    Output: 
        (1) inputs dict:
            seq-encoder: src_seq, lengths, feat_seqs
            graph-encoder: edges, mask, type, feat_seqs (, adjacent_matrix)
            encoder-transform: index, lengths, root, node_lengths
            decoder: tgt_seq, src_seq, feat_seqs
            decoder-transform: index
        (2) (max_node_num, max_node_size)
        (3) (generation, classification)
        (4) (copy_gold, copy_switch)
    """
    # print("测试####################################")
    # print("batch",batch)
    # assert 0==1

    inputs = {'seq-encoder':{}, 'graph-encoder':{},'decoder':{}, 'decoder-transform':{}}
    ###===== RNN encoder =====###
    src_seq,src_mask, tgt_seq, tgt_seq_mask = batch['src'][0], batch['src_mask'], batch['tgt'], batch['tgt_mask']
    label_tgt_ans= batch['label_tgt_ans']
    # print('src_seq',src_seq.shape)
    # print('src_mask',src_mask.shape)

    # print('<?????>::', tokenizer.decode(src_seq[0]))
    # print('<?????>::', tokenizer.decode(tgt_seq[0]))
    # print('src_seq',src_seq.shape)
    # print('label_tgt_ans',label_tgt_ans.shape)

    # print('src_mask',src_mask.shape)
    # print('src_mask',src_mask)
    inputs['seq-encoder']['src_seq'] = src_seq
    inputs['seq-encoder']['src_mask'] = src_mask
    inputs['seq-encoder']['ans_seq'], inputs['seq-encoder']['ans_lengths'] = batch['ans'][0], batch['ans'][1]
    inputs['seq-encoder']['ans_mask'] = batch['ans_mask']

    classification= label_tgt_ans

    inputs['decoder']['tgt_seq'] = tgt_seq
    # print('tgt_seq.shape',tgt_seq.shape)
    max_tgt_seq_length=tgt_seq.shape[1]
    inputs['decoder']['tgt_seq_mask'] = tgt_seq_mask
    # tgt_seq_length = torch.sum(tgt_seq_mask, 1)
    # # print('tgt_seq_length',tgt_seq_length)
    # max_tgt_seq_length = torch.max(tgt_seq_length)
    inputs['max_tgt_seq_length'] = max_tgt_seq_length
    generation=tgt_seq
    # print('max_tgt_seq_length',max_tgt_seq_length)
    # generation_mask = tgt_seq_mask[:, :max_tgt_seq_length - 2]
    # generation = tgt_seq[:, :max_tgt_seq_length - 2]
    # print('<?????>::', tokenizer.decode(generation[0]))
    # assert 0==1

    # src_length=torch.sum(src_mask,1)
    # # print('src_length',src_length)
    # # assert 0==1
    #
    # max_src_length=torch.max(src_length)
    # # print('src_seq.shape',src_seq.shape)
    #
    #
    #
    # # print('max_src_length',max_src_length)
    # src_seq1=src_seq[:,:max_src_length]
    # print(src_seq1)
    # print('<?????>::',tokenizer.decode(src_seq1[0]))
    # # assert 0==1
    # # print('_______',src_seq1.shape)
    # src_mask=src_mask[:,:max_src_length]
    #
    # inputs['seq-encoder']['src_mask'] = src_mask
    # # print('+++++++',src_mask.shape)
    #
    # # print('tgt_seq_mask',tgt_seq_mask)
    # # print('tgt_seq',tgt_seq)
    # # print('tgt_seq',tgt_seq.shape)
    # tgt_seq_length=torch.sum(tgt_seq_mask,1)
    # # print('tgt_seq_length',tgt_seq_length)
    # max_tgt_seq_length=torch.max(tgt_seq_length)
    # # print('max_tgt_seq_length', max_tgt_seq_length)
    # tgt_seq=tgt_seq[:,:max_tgt_seq_length]
    # print(' tgt_seq', tgt_seq.shape)
    # print('<?????>::', tokenizer.decode( tgt_seq[0]))
    # # assert 0==1
    #
    # tgt_seq_mask=tgt_seq_mask[:,:max_tgt_seq_length]
    # # print('tgt_seq',tgt_seq.shape)
    # inputs['decoder']['tgt_seq'] = tgt_seq
    # inputs['decoder']['tgt_seq_mask']= tgt_seq_mask
    # inputs['max_tgt_seq_length']= max_tgt_seq_length
    # # src_seq, lengths = src_seq[0], src_seq[1]
    # inputs['seq-encoder']['src_seq']=src_seq1
    # # print('<<<>>>',inputs['seq-encoder']['src_seq'].shape)
    # inputs['seq-encoder']['lengths'] = src_length
    # inputs['seq-encoder']['src_mask']=src_mask
    # inputs['seq-encoder']['ans_seq'], inputs['seq-encoder']['ans_lengths'] = batch['ans'][0], batch['ans'][1]
    # inputs['seq-encoder']['ans_mask']=batch['ans_mask']
    ###===== 获取依存关系矩阵和二值掩码矩阵 =====###
    edges = batch['edges']
    # print('edges',edges)
    graph_undir=get_edges(edges[0])
    # print('graph_undir.shape',graph_undir.shape)
    binary_edge_mask=get_edges(edges[1])
    # print('binary_edge_mask.shape',binary_edge_mask.shape)
    inputs['graph-encoder']["graph_undir"] = graph_undir
    # print('graph_undir[1][0].shape',graph_undir[1][0].shape)
    inputs['graph-encoder']["edge_matrix_undir"] = binary_edge_mask

    # generation_mask = tgt_seq_mask[:, :max_tgt_seq_length-2]
    # generation = tgt_seq[:, :max_tgt_seq_length-2]

    # gen_length = torch.sum(generation_mask, 1)
    # max_gen_length=torch.max(gen_length)
    # generation=tgt_seq[:,:max_tgt_seq_length-1]
    # print('generation',generation)
    # inputs['decoder']['src_seq'] = src_seq  # nodes
    # inputs['decoder']['ans_seq'] = batch['ans'][0]
    ###===== decoder =====###
    # print('tgt_seq',tgt_seq)
    # print('tgt_seq',tgt_seq.shape)
    # generation = tgt_seq[:, 1:]   # exclude [BOS] token

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # print('decoder input:::',tokenizer.decode(tgt_seq[0].detach().cpu().numpy()))
    # print('target:::',tokenizer.decode(generation[0].detach().cpu().numpy()))

   
    copy_gold, copy_switch = None, None
    if copy:
        copy_gold, copy_switch = batch['copy'][1], batch['copy'][0]
        copy_gold, copy_switch = copy_gold[:, 1:], copy_switch[:, 1:]

    # return inputs, generation
    return inputs, generation,classification

    # return inputs, (max_node_num, max_node_size), (generation, classification), (copy_gold, copy_switch)
