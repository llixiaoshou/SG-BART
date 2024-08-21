from __future__ import division
import math
import random
import numpy as np
import torch
from torch import cuda
import onqg.dataset.Constants as Constants
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-base')

class Dataset(object):

    def __init__(self, seq_datasets, graph_datasets, batchSize, 
                 copy=False, answer=False, ans_feature=False, 
                 feature=False, node_feature=False, opt_cuda=False):

        # print('seq_datasets',seq_datasets)
        self.src, self.tgt ,self.ans,self.label_tgt_ans= seq_datasets['src'], seq_datasets['tgt'],seq_datasets['ans'],seq_datasets['label_tag_ans']
        # self.src, self.tgt, self.ans = seq_datasets['src'], seq_datasets['tgt'], seq_datasets[
        #     'ans']

        self.src_attention_mask,self.tgt_attention_mask,self.ans_attention_mask = seq_datasets['src_mask'], seq_datasets['tgt_mask'], seq_datasets['ans_mask']
        self.has_tgt = True if self.tgt else False
        # self.graph_index = graph_datasets['index']

        # print('self.label_tgt_ans',self.label_tgt_ans)
        # assert 0==1

        self.edges= graph_datasets['edge']['matrix_undir']
        self.binary_edge_mask=graph_datasets['edge']['binary_edge_mask']

        self.answer = answer
        self.ans = seq_datasets['ans'] if answer else None
        self.ans_feature_num = len(seq_datasets['ans_feature']) if ans_feature else 0
        self.ans_features = seq_datasets['ans_feature'] if self.ans_feature_num else None
        self.feature_num = len(seq_datasets['feature']) if feature else 0
        self.features = seq_datasets['feature'] if self.feature_num else None

        # self.node_feature_num = len(graph_datasets['feature']) if node_feature else 0
        self.node_feature_num =  0
        # self.node_features = graph_datasets['feature'] if self.node_feature_num else None
        self.node_features=None

        self.copy = copy
        self.copy_switch = seq_datasets['copy']['switch'] if copy else None
        # self.copy_tgt = seq_datasets['copy']['tgt'] if copy else None
        
        self._update_data()
        
        if opt_cuda:
            cuda.set_device(opt_cuda[0])
        self.device = torch.device("cuda" if cuda else "cpu")
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)
        # self.idxs = list(range(len(self.src)))
    
    def _update_data(self):
        """sort all data by lengths of source text"""

        self.idxs = list(range(len(self.src)))
        print('++++++原序号',self.idxs)
        lengths = [s.size(0) for s in self.src]
        RAW = [lengths, self.src, self.idxs]
        DATA = list(zip(*RAW))
        DATA.sort(key=lambda x:x[0])
        self.src = [d[1] for d in DATA]
        self.idxs = [d[2] for d in DATA]

        # print('self.idxs',self.idxs)
        if self.src_attention_mask:
            self.src_attention_mask = [self.src_attention_mask[idx] for idx in self.idxs]

        if self.tgt:
            self.tgt = [self.tgt[idx] for idx in self.idxs]
        if self.tgt_attention_mask:
            self.tgt_attention_mask = [self.tgt_attention_mask[idx] for idx in self.idxs]
        if self.copy:
            self.copy_switch = [self.copy_switch[idx] for idx in self.idxs]
            self.copy_tgt = [self.copy_tgt[idx] for idx in self.idxs]
        if self.feature_num:
            self.features = [[feature[idx] for idx in self.idxs] for feature in self.features]
        if self.answer:
            self.ans = [self.ans[idx] for idx in self.idxs]
        if self.ans_attention_mask:
            self.ans_attention_mask = [self.ans_attention_mask[idx] for idx in self.idxs]
            if self.ans_feature_num:
                self.ans_features = [[feature[idx] for idx in self.idxs] for feature in self.ans_features]
        if self.label_tgt_ans:
            self.label_tgt_ans = [self.label_tgt_ans[idx] for idx in self.idxs]

        self.edges_dict = self._get_edge_dict(self.edges)
        self.binary_edge_mask_dict = self._get_edge_dict(self.binary_edge_mask)


    def _get_edge_dict(self, edges):
        # print("edges",edges)
        edges_dict = []
        for sample in edges:
            edge_dict = []
            for edge_list in sample:
                edge_dict.append([(idx, edge.item()) for idx, edge in enumerate(edge_list) if edge.item() != Constants.PAD])
            edges_dict.append(edge_dict)
        # print(edges_dict)
        return edges_dict

    def _batchify(self, data, align_right=False, include_lengths=False, src_len=None, need_mask=False):
        """get data in a batch while applying padding, return length if needed"""
        if src_len:
            lengths = src_len
        else:
            lengths = [x.size(0) for x in data]
        # print('lengths',lengths)
        max_length = max(lengths)
        # print('max_length',max_length)
        out = data[0].new(len(data), max_length).fill_(tokenizer.pad_token_id)
        # print('out',out)
        # if need_mask:
            # print('out', out.size())
        # mask_
        mask_ = torch.ones(out.size())
        # print('out',out.shape)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
            data_length = data[i].size(0)
            mask_[i][data_length: max_length] = 0
        # print('out+++',out)
        # if need_mask:
        #     print('mask_', mask_)
        #     print('mask_', mask_.size())
        if include_lengths and need_mask:
            return out, lengths, mask_
        elif need_mask:
            return out, mask_
        elif include_lengths:
            return out, lengths
        else:
            return out
    
    def _graph_length_info(self, edges):
        # indexes = [index for sample in indexes for index in sample]
        # index_length = [len(index) for index in indexes]

        node_length = [len(edge) for edge in edges]
        nodes = [torch.Tensor([i for i in range(length)]) for length in node_length]
        tmpNodesBatch = self._batchify(nodes, src_len=node_length)
        return  node_length, tmpNodesBatch



    def _pad_edges(self, edges, node_length, align_right=False):
        max_length = max(node_length)
        edge_lengths = []
        outs = []
        for edge, data_length in zip(edges, node_length):
            out = edge[0].new_full((max_length, max_length), Constants.PAD)
            for i, e in enumerate(edge):
                # print('e',e)
                # print('len(e)',len(e))
                offset = max_length - data_length if align_right else 0
                # print('data_length',data_length)
                out[i].narrow(0, offset, data_length).copy_(e)
            outs.append(out.view(-1))
            edge_lengths.append(data_length * data_length)
        outs = torch.stack(outs, dim=0)     # batch_size x (max_length * max_length)

        # print('_______',outs)

        return outs, edge_lengths

    def __getitem__(self, index):
        """get the exact batch using index, and transform data into Tensor form"""

        # print('______________',self.numBatches)
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        # print('index',index)
        # print('++++++++++++',self.batchSize)
        srcBatch, lengths = self._batchify(self.src[index * self.batchSize: (index + 1) * self.batchSize], align_right=False, include_lengths=True,need_mask=False)
        # print('srcBatch.shape',srcBatch.shape)

        # print('src_attention_mask',self.src_attention_mask)
        # assert 0==1
        #
        src_mask = self._batchify(self.src_attention_mask[index * self.batchSize: (index + 1) * self.batchSize],align_right=False, include_lengths=False, need_mask=False)
        # print('src_mask.shape',src_mask.shape)
        # assert 0==1
        # # tgtBatch = None
        if self.tgt:
            tgtBatch = self._batchify(self.tgt[index * self.batchSize: (index + 1) * self.batchSize], need_mask=False)
            tgt_mask = self._batchify(self.tgt_attention_mask[index * self.batchSize: (index + 1) * self.batchSize],need_mask=False)
            # print('tgt_mask', tgt_mask)

        idxBatch = self.idxs[index * self.batchSize: (index + 1) * self.batchSize]
        label_tgt_ans_batch=self._batchify(self.label_tgt_ans[index * self.batchSize: (index + 1) * self.batchSize],need_mask=False)
        # print('label_tgt_ans_batch',label_tgt_ans_batch)

        # if self.node_feature_num:
        #     nodeFeatBatches = [
        #         self._batchify([feat[i] for i in idxBatch], src_len=node_lengths) for feat in self.node_features
        #     ]

        # print('idxBatch',idxBatch)
        # graphIndexBatch = [self.graph_index[i] for i in idxBatch]
        # graphRootBatch = [self.graph_root[i] for i in idxBatch]

        #增加边的信息
        edgesBatch= [self.edges[i] for i in idxBatch]
        edgesDict = [self.edges_dict[i] for i in idxBatch]
        binary_edge_mask_Batch = [self.binary_edge_mask[i] for i in idxBatch]
        binary_edge_mask_Batch_Dict = [self.binary_edge_mask_dict[i] for i in idxBatch]
        #增加节点的信息
        node_lengths, tmpNodesBatch = self._graph_length_info(edgesBatch)
        #增加边的信息
        # print('edgesBatch', len(edgesBatch))
        # print('node_lengths',node_lengths)
        edgesBatch, edges_lengths = self._pad_edges(edgesBatch, node_lengths)
        # print('edges_lengths',edges_lengths)
        # print('edgesBatch',edgesBatch.shape)
        # assert 0==1
        binary_edge_mask_Batch, _= self._pad_edges(binary_edge_mask_Batch, node_lengths)
        # print('binary_edge_mask_Batch',binary_edge_mask_Batch)


        # nodeFeatBatches = None
        # if self.node_feature_num:
        #     nodeFeatBatches = [
        #         self._batchify([feat[i] for i in idxBatch], src_len=node_lengths) for feat in self.node_features
        #     ]
        
        featBatches = None
        if self.feature_num:
            featBatches = [
                self._batchify(feat[index * self.batchSize: (index + 1) * self.batchSize], 
                               src_len=lengths) for feat in self.features
            ]
        
        copySwitchBatch, copyTgtBatch = None, None
        if self.copy:
            copySwitchBatch = self._batchify(self.copy_switch[index * self.batchSize: (index + 1) * self.batchSize])
            copyTgtBatch = self._batchify(self.copy_tgt[index * self.batchSize: (index + 1) * self.batchSize])
        
        ansBatch, ansFeatBatches = None, None
        if self.answer:
            ansBatch, ansLengths = self._batchify(self.ans[index * self.batchSize: (index + 1) * self.batchSize],
                                                  align_right=False, include_lengths=True,need_mask=False)
            ans_mask = self._batchify(self.ans_attention_mask[index * self.batchSize: (index + 1) * self.batchSize],align_right=False, include_lengths=False, need_mask=False)
            if self.ans_feature_num:
                ansFeatBatches = [
                    self._batchify(feat[index * self.batchSize: (index + 1) * self.batchSize], src_len=ansLengths) 
                    for feat in self.ans_features
                ]

        def wrap(b):
            if b is None:
                return b
            b = torch.stack([x for x in b], dim=0).contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1).to(self.device)
        edge_lengths = torch.LongTensor(edges_lengths).view(1, -1).to(self.device)

        # print('edge_lengths',edge_lengths)
        # assert 0==1
        # indices = range(len(srcBatch))
        indices = range(len( self.src))
        
        rst = {}
        rst['indice'] = indices
        # print('srcBatch',srcBatch.shape)
        rst['src'] = (wrap(srcBatch), lengths)
        # rst['src'] = (wrap(self.src), lengths)
        rst['raw-index'] = idxBatch
        rst['src_mask']=src_mask
        # rst['src_mask'] =self.src_attention_mask
        if self.has_tgt:
            rst['tgt'] = wrap(tgtBatch)
            # rst['tgt'] = wrap(self.tgt)
            rst['tgt_mask'] = tgt_mask
            # rst['tgt_mask'] = wrap(self.tgt_attention_mask)
        if self.copy:
            rst['copy'] = (wrap(copySwitchBatch), wrap(copyTgtBatch))
        if self.answer:
            ansLengths = torch.LongTensor(ansLengths).view(1, -1).to(self.device)
            rst['ans'] = (wrap(ansBatch), ansLengths)
            rst['ans_mask']=ans_mask
            # rst['ans'] = (wrap(self.ans), ansLengths)
            # rst['ans_mask'] = self.ans_attention_mask
            if self.ans_feature_num:
                rst['ans_feat'] = (tuple(wrap(x) for x in ansFeatBatches), ansLengths)
        if self.feature_num:
            rst['feat'] = (tuple(wrap(x) for x in featBatches), lengths)


        rst['label_tgt_ans'] = wrap(label_tgt_ans_batch)


        #增加边的信息

        m=wrap(edgesBatch)
        # print(m)
        # assert 0==1
        # n=wrap(binary_edge_mask_Batch)
        # print('edgesBatch',edgesBatch.shape)
        # print('binary_edge_mask_Batch',binary_edge_mask_Batch.shape)
        # assert 0==1
        rst['edges'] = (wrap(edgesBatch),wrap(binary_edge_mask_Batch),edge_lengths)
        # tmp1= rst['edges'][0]
        # tmp2=tmp1[0]
        # rst['edges_dict'] = (edgesDict)
        # if self.node_feature_num:
        #     rst['node_feat'] = (tuple(wrap(x) for x in nodeFeatBatches), node_lengths)
        # rst["graph_undir"] = m.reshape(len(node_lengths),max(node_lengths),max(node_lengths))
        # rst["binary_edge_mask"] = n.reshape(len(node_lengths),max(node_lengths),max(node_lengths))
        return rst

    def __len__(self):
        return self.numBatches

    # def shuffle(self):
    #     """shuffle the order of data in every batch"""
    #     def shuffle_group(start, end, NEW):
    #         # print('***')
    #         """shuffle the order of samples with index from start to end"""
    #         # print('start, end, NEW', start, end, NEW)
    #         RAW = [self.src[start:end], self.tgt[start:end], self.idxs[start:end]]
    #         DATA = list(zip(*RAW))
    #         index = torch.randperm(len(DATA))
    #         src, tgt, idx = zip(*[DATA[i] for i in index])
    #         NEW['SRCs'] += list(src)
    #         NEW['TGTs'] += list(tgt)
    #         NEW['IDXs'] += list(idx)
    #         if self.answer:
    #             ans = [self.ans[start:end][i] for i in index]
    #             NEW['ANSs'] += ans
    #             if self.ans_feature_num:
    #                 ansft = [[feature[start:end][i] for i in index] for feature in self.ans_features]
    #                 for i in range(self.ans_feature_num):
    #                     NEW['ANSFTs'][i] += ansft[i]
    #
    #         if self.feature_num:
    #             ft = [[feature[start:end][i] for i in index] for feature in self.features]
    #             for i in range(self.feature_num):
    #                 NEW['FTs'][i] += ft[i]
    #
    #         if self.copy:
    #             cpswt = [self.copy_switch[start:end][i] for i in index]
    #             cptgt = [self.copy_tgt[start:end][i] for i in index]
    #             NEW['COPYSWTs'] += cpswt
    #             NEW['COPYTGTs'] += cptgt
    #
    #         return NEW
    #
    #     assert self.tgt != None, "shuffle is only aimed for training data (with target given)"
    #
    #     NEW = {'SRCs':[], 'TGTs':[], 'IDXs':[]}
    #     if self.copy:
    #         NEW['COPYSWTs'], NEW['COPYTGTs'] = [], []
    #     if self.feature_num:
    #         NEW['FTs'] = [[] for i in range(self.feature_num)]
    #     if self.answer:
    #         NEW['ANSs'] = []
    #         if self.ans_feature_num:
    #             NEW['ANSFTs'] = [[] for i in range(self.ans_feature_num)]
    #
    #     shuffle_all = random.random()
    #     if shuffle_all > 0.75:  # fix this magic number later
    #         # print('if')
    #         start, end = 0, self.batchSize * self.numBatches
    #         NEW = shuffle_group(start, end, NEW)
    #     else:
    #         for batch_idx in range(self.numBatches):
    #             # print('else')
    #             start = batch_idx * self.batchSize
    #             end = start + self.batchSize
    #             NEW = shuffle_group(start, end, NEW)
    #
    #     self.src, self.tgt, self.idxs = NEW['SRCs'], NEW['TGTs'], NEW['IDXs']
    #     if self.copy:
    #         self.copy_switch, self.copy_tgt = NEW['COPYSWTs'], NEW['COPYTGTs']
    #     if self.answer:
    #         self.ans = NEW['ANSs']
    #         if self.ans_feature_num:
    #             self.ans_features = NEW['ANSFTs']
    #     if self.feature_num:
    #         self.features = NEW['FTs']


