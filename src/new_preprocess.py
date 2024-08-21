import math

import numpy as np
import spacy
import json
import codecs
import torch
import argparse
from tqdm import tqdm
import pargs
import onqg.dataset.Constants as Constants
from onqg.dataset import Vocab
from onqg.dataset import Dataset
import copy
# from transformers import AutoTokenizer
from transformers import BartTokenizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
token_nize_for_tokenize = spacy.load('en_core_web_trf')
tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-large')
json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    data = data.split('\n')
    data = [sent.replace("\u200e", "").split(' ') for sent in data]
    # print('len(data)',len(data))
    # data = [sent.split(' ') for sent in data]
    # print('+++++',len(data))
    # assert 0==1
    return data

def filter_data(data_list, opt):
    data_num = len(data_list)
    # print('++++++++@@@@@@@@@',data_num)
    # assert 0==1
    idx = list(range(len(data_list[0])))
    # print('idx',idx)
    # assert 0==1
    rst = [[] for _ in range(data_num)]
    # final_indexes = []
    print('+++++++#############',len(data_list[0]))
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        src_len = len(src)
        # print('src_len',src_len)
        # assert 0==1
        if src_len <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length:
            if len(src) * len(tgt) > 0 and src_len >= 10:  # TODO: fix this magic number
                rst[0].append(src)
                # print('len(rst[0])',len(rst[0]))
                # rst[1].append([Constants.BOS_WORD] + tgt + [Constants.EOS_WORD])
                rst[1].append(tgt)
                # final_indexes.append(i)
                for j in range(2, data_num):
                    sent = data_list[j][i]
                    rst[j].append(sent)

    print("change data size from " + str(len(idx)) + " to " + str(len(rst[0])))
    final_indexes = [_ for _ in range(len(rst[0]))]


    # print('final_indexes',final_indexes)
    # print('len(rst[0])',len(rst[0]))
    # assert 0==1
    return rst, final_indexes

def get_data(files, opt):
    # print('files',files)
    # assert 0==1
    src, tgt = load_file(files['src'])[0:2],load_file(files['tgt'])[0:2]
    # print('src',src)
    # print('tgt',tgt)
    # src, tgt = load_file(files['src']), load_file(files['tgt'])
    data_list = [src, tgt]
    if opt.answer:
        data_list.append(load_file(files['ans'])[0:2])
    if opt.feature:
        data_list +=[load_file(filename) for filename in files['feats']]

    # idx = list(range(len(data_list[0])))

    data_list, final_indexes = filter_data(data_list, opt)
    # assert 0==1

    # print('data_list',data_list[0])
    # print('data_list+++',data_list[1])

    rst = {'src': data_list[0], 'tgt': data_list[1]}
    i = 2
    if opt.answer:
        rst['ans'] = data_list[i]
        i += 1
    if opt.feature:
        rst['feats'] = [data_list[i] for i in range(i, i + len(files['feats']))]
        i += 1
    # print('final_indexes',final_indexes)
    # assert 0==1
    return rst, final_indexes

def merge(src_input_ids,ans_input_ids,src_attention_masks,ans_attention_masks):
    result=[]
    result_attention_mask=[]
    result_envi=[]
    for src_ids,ans_ids,src_mask,ans_mask in zip(src_input_ids,ans_input_ids, src_attention_masks,ans_attention_masks):
        # print('src_ids',src_ids)
        # print('ans_ids',ans_ids)
        # print('src_mask',src_mask)
        # print('ans_mask',ans_mask)
        merged_tensor = torch.cat((src_ids, ans_ids[1:]))
        merged_mask_tensor=torch.cat((src_mask,ans_mask[1:]))
        result.append(merged_tensor)
        result_attention_mask.append(merged_mask_tensor)

    return result,result_attention_mask

def text_split(text):
    text = [w for w in text if w != ""]
    m = " ".join(text)
    document = token_nize_for_tokenize(m)
    n = []
    for token in document:
        n.append(token.text)
    return n

def single_tokenize(token):
    acc = ' ' + token
    token_wordpice = tokenizer(acc, add_special_tokens=False)
    w_len = len(token_wordpice["input_ids"])
    return w_len

def tag_ans_envi_label(text, text_tgt,text_ans):
    # return labels
    src = text_split(text)
    # print(src)
    # assert 0 == 1
    tgt = text_split(text_tgt)
    ans = text_split(text_ans)
    tgt_ans = tgt+ans
    src_sen = " ".join(src)
    words_pieces = tokenizer(src_sen, add_special_tokens=False)
    words_pieces_ids = words_pieces["input_ids"]
    # print('len(words_pieces_ids)',len(words_pieces_ids))

    pattern = re.compile('[\W]*')
    label = [0 for _ in range(len(words_pieces_ids))]
    porter_stemmer, stopwords_eng = PorterStemmer(), stopwords.words('english')
    src1 = [porter_stemmer.stem(_) for _ in src]
    tgt_ans1 = [porter_stemmer.stem(_) for _ in tgt_ans]

    for index, src_token in enumerate(src1):
        if len(src_token) > 0 and src_token not in stopwords_eng and not pattern.fullmatch(src_token):
            if src_token in tgt_ans1:
                if index==0:
                    token_wordpice = tokenizer(src[index], add_special_tokens=False)
                    length = len(token_wordpice["input_ids"])
                else:
                    length = single_tokenize(src[index])
                # print('length',length)
                start = index
                # print('start',start)
                end = start + length
                # print('end',end)
                for i in range(start, end):
                    label[i] = 1
    # print(label)
    # print(label)
    # print(src)
    # print(tgt_ans)
    # assert 0 == 1
    # print('label', len(label))
    # label.insert(0,0) #在首部插入0
    # label.append(0) # 在尾部插入0
    # # print('label',len(label))
    # # print('label',label)
    # # assert 0==1
    # label_tensor = torch.tensor(label)
    return label


def tag_envi_label(text, text_tgt):
    # return labels
    src = text_split(text)
    # print(src)
    # assert 0 == 1
    tgt = text_split(text_tgt)
    # ans = text_split(text_ans)
    tgt_ans = src+tgt
    src_sen = " ".join(src)
    words_pieces = tokenizer(src_sen, add_special_tokens=False)
    words_pieces_ids = words_pieces["input_ids"]
    # print('len(words_pieces_ids)',len(words_pieces_ids))

    pattern = re.compile('[\W]*')
    label = [0 for _ in range(len(words_pieces_ids))]
    porter_stemmer, stopwords_eng = PorterStemmer(), stopwords.words('english')
    src1 = [porter_stemmer.stem(_) for _ in src]
    tgt_ans1 = [porter_stemmer.stem(_) for _ in tgt_ans]

    for index, src_token in enumerate(src1):
        if len(src_token) > 0 and src_token not in stopwords_eng and not pattern.fullmatch(src_token):
            if src_token in tgt_ans1:
                if index==0:
                    token_wordpice = tokenizer(src[index], add_special_tokens=False)
                    length = len(token_wordpice["input_ids"])
                else:
                    length = single_tokenize(src[index])
                # print('length',length)
                start = index
                # print('start',start)
                end = start + length
                # print('end',end)
                for i in range(start, end):
                    label[i] = 1
    # print(label)
    # print(label)
    # print(src)
    # print(tgt_ans)
    # assert 0 == 1
    # print('label', len(label))
    # label.insert(0,0) #在首部插入0
    # label.append(0) # 在尾部插入0
    # # print('label',len(label))
    # # print('label',label)
    # # assert 0==1
    # label_tensor = torch.tensor(label)
    return label

def sequence_data(opt):
    # ========== get data ==========#
    train_files = {'src': opt.train_src, 'tgt': opt.train_tgt}
    valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt}

    if opt.answer:
        assert opt.train_ans and opt.valid_ans, "Answer files of train and valid must be given"
        train_files['ans'], valid_files['ans'] = opt.train_ans, opt.valid_ans
    if opt.feature:
        assert len(opt.train_feats) == len(opt.valid_feats) and len(opt.train_feats) > 0
        train_files['feats'], valid_files['feats'] = opt.train_feats, opt.valid_feats

    train_data, train_final_indexes = get_data(train_files, opt)
    valid_data, valid_final_indexes = get_data(valid_files, opt)
    train_src, train_tgt = train_data['src'], train_data['tgt']
    # print('train_src',train_src)
    # print(len(train_src[0]))
    # print('train_tgt',train_tgt)
    valid_src, valid_tgt= valid_data['src'], valid_data['tgt']

    train_ans = train_data['ans'] if opt.answer else None
    # print(' train_ans', train_ans)
    valid_ans = valid_data['ans'] if opt.answer else None
    train_feats = train_data['feats'] if opt.feature else None
    valid_feats = valid_data['feats'] if opt.feature else None
    train_envi=[]
    for s, t, a in zip(train_src, train_tgt, train_ans):
        src_envi = tag_ans_envi_label(s, t,a)
        src_envi.insert(0, 0)  # 在首部插入0
        src_envi.append(0)  # 在尾部插入0
        ans_envi=tag_envi_label(a, t)
        ans_envi.append(0)  # 在尾部插入0
        envi = src_envi+ans_envi
        envi = torch.tensor(envi)
        train_envi.append(envi)
    # print('train_envi',train_envi)
    valid_envi=[]
    for s, t, a in zip(valid_src, valid_tgt, valid_ans):
        src_envi = tag_ans_envi_label(s, t,a)
        src_envi.insert(0, 0)  # 在首部插入0
        src_envi.append(0)  # 在尾部插入0
        ans_envi = tag_envi_label(a, t)
        ans_envi.append(0)  # 在尾部插入0
        envi = src_envi + ans_envi
        envi = torch.tensor(envi)
        valid_envi.append(envi)
    # print('valid_envi',valid_envi)

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # tokenizer = AutoTokenizer.from_pretrained('../pre_model/bart-base')

    # tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-base')

    # ========== build vocabulary ==========#
    print('Loading pretrained word embeddings ...')
    print('Done .')
    print("build src & tgt vocabulary")
    corpus = train_src + train_tgt

    options = {'lower': True, 'mode': opt.vocab_trunc_mode, 'tgt': True,
               'size': max(opt.src_vocab_size, opt.tgt_vocab_size),
               'frequency': min(opt.src_words_min_frequency, opt.tgt_words_min_frequency)}

    vocab = Vocab.from_opt(corpus=corpus, opt=options)

    src_vocab = tgt_vocab = vocab
    options = {'lower': False, 'mode': 'size', 'size': opt.feat_vocab_size,
               'frequency': opt.feat_words_min_frequency, 'tgt': False}

    feats_vocab = [Vocab.from_opt(corpus=feat, opt=options) for feat in train_feats] if opt.feature else None

    ####进行分词####
    train_src_input_ids, train_src_attention_masks, train_src_tokens = [],[],[]
    for train_tokens_input in train_src:
        text = [w for w in train_tokens_input if w != ""]
        m = " ".join(text)
        document = token_nize_for_tokenize(m)
        n = []
        for token in document:
            n.append(token.text)
        # print('n',n)
        # assert 0 == 1
        # token_new = n
        # train_tokens = tokenizer(" ".join(train_tokens_input), return_tensors='pt', max_length=200, padding="max_length", truncation=True)
        train_tokens = tokenizer(" ".join(n), return_tensors='pt')
        input_id, attention_mask = train_tokens["input_ids"][0], train_tokens['attention_mask'][0]
        src_length =torch.sum(attention_mask==1)
        train_src_input_ids.append(input_id)
        # print('train_src_input_ids',train_src_input_ids)
        #
        # print('train_attenton_mask',attention_mask)
        train_src_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_src_tokens.append(temp)


    valid_src_input_ids, valid_src_attention_masks, valid_src_tokens = [], [], []
    for valid_tokens_input in valid_src:
        text = [w for w in valid_tokens_input if w != ""]
        m = " ".join(text)
        document = token_nize_for_tokenize(m)
        n = []
        for token in document:
            n.append(token.text)
        valid_tokens = tokenizer(" ".join(n), return_tensors='pt')
        # valid_tokens = tokenizer(" ".join(valid_tokens_input), return_tensors='pt', max_length=200, padding="max_length",
        #                                 truncation=True)
        input_id, attention_mask = valid_tokens["input_ids"][0], valid_tokens['attention_mask'][0]
        valid_src_input_ids.append(input_id)
        valid_src_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_src_tokens.append(temp)


    train_tgt_input_ids, train_tgt_attention_masks,train_tgt_tokens = [],[],[]
    for train_tokens_input in train_tgt:
        train_tgt_token = tokenizer(" ".join(train_tokens_input), return_tensors='pt')
        input_id, attention_mask = train_tgt_token["input_ids"][0], train_tgt_token['attention_mask'][0]
        train_tgt_input_ids.append(input_id)
        train_tgt_attention_masks.append(attention_mask)

        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_tgt_tokens.append(temp)

    valid_tgt_input_ids, valid_tgt_attention_masks,valid_tgt_tokens = [], [], []
    for valid_tokens_input in valid_tgt:
        valid_tgt_token = tokenizer(" ".join(valid_tokens_input), return_tensors='pt')
        input_id, attention_mask = valid_tgt_token["input_ids"][0], valid_tgt_token['attention_mask'][0]
        valid_tgt_input_ids.append(input_id)
        valid_tgt_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_tgt_tokens.append(temp)

    # ans 分词（BART分词）
    train_ans_input_ids, train_ans_attention_masks, train_ans_tokens = [], [], []
    for train_ans_token_input in train_ans:
        text = [w for w in train_ans_token_input if w != ""]
        m = " ".join(text)
        document = token_nize_for_tokenize(m)
        n = []
        for token in document:
            n.append(token.text)
        train_ans_token = tokenizer(" ".join(n), return_tensors='pt')
        input_id, attention_mask = train_ans_token["input_ids"][0], train_ans_token['attention_mask'][0]
        # print('input_id',input_id)
        # assert 0==1
        train_ans_input_ids.append(input_id)
        train_ans_attention_masks.append(attention_mask)
        # print('train_ans_input_ids',train_ans_input_ids)
        # assert 0==1
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_ans_tokens.append(temp)

    valid_ans_input_ids, valid_ans_attention_masks, valid_ans_tokens = [], [], []
    for valid_ans_token_input in valid_ans:
        text = [w for w in valid_ans_token_input if w != ""]
        m = " ".join(text)
        document = token_nize_for_tokenize(m)
        n = []
        for token in document:
            n.append(token.text)
        valid_ans_token = tokenizer(" ".join(n), return_tensors='pt')
        input_id, attention_mask = valid_ans_token["input_ids"][0], valid_ans_token['attention_mask'][0]
        valid_ans_input_ids.append(input_id)
        valid_ans_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_ans_tokens.append(temp)

    train_result,train_result_attention_masks=merge(train_src_input_ids,train_ans_input_ids,train_src_attention_masks,train_ans_attention_masks)


    valid_result,valid_result_attention_masks=merge(valid_src_input_ids,valid_ans_input_ids,valid_src_attention_masks,valid_ans_attention_masks)

    # print('train_result',len(train_result[0]))
    # print('train_envi',len(train_envi[0]))


    data = {'settings': opt,
            'dict': {'src': src_vocab,
                     'tgt': tgt_vocab,
                     },
            'train': {'src':train_result ,
                      'tgt': train_tgt_input_ids,
                      'ans': train_ans_input_ids,
                      'label_tag_ans':train_envi,
                      # 'src_ans': train_result,
                      # 'feature': train_feat_idxs,
                      'src_mask':train_result_attention_masks,
                      'tgt_mask':train_tgt_attention_masks,
                      'ans_mask':train_ans_attention_masks,
                      },
            'valid': {'src': valid_result,
                      'tgt': valid_tgt_input_ids,
                      'ans': valid_ans_input_ids,
                      'label_tag_ans':valid_envi,
                      # 'src_ans':valid_result,
                      # 'feature': valid_feat_idxs,
                      'src_mask': valid_result_attention_masks,
                      'tgt_mask': valid_tgt_attention_masks,
                      'ans_mask': valid_ans_attention_masks,
                      'tokens': {'src': valid_src_tokens,
                                 'tgt': valid_tgt_tokens}
                      }
            }
    # print('data',data)
    # assert 0==1
    return data, (train_final_indexes, valid_final_indexes), (None, None)

def convert_word_to_idx(text, vocab, lower=True, sep=False, pretrained=''):
    def lower_sent(sent):
        for idx, w in enumerate(sent):
            if w not in [Constants.BOS_WORD, Constants.EOS_WORD, Constants.PAD_WORD, Constants.UNK_WORD,
                         Constants.SEP_WORD]:
                sent[idx] = w.lower()
        return sent

    def get_dict(sent, length, raw, separate=False):
        raw = raw.split(' ')
        bert_sent = [w for w in sent]
        ans_indexes = []
        if separate:
            sep_id = sent.index(Constants.SEP_WORD)
            sent = sent[:sep_id]
            ans_indexes = [i for i in range(sep_id + 1, len(bert_sent) - 1)]

        indexes = [[i for i in ans_indexes] for _ in raw]
        word, idraw = '', 0
        for idx, w in enumerate(sent):
            if word == raw[idraw] and idx != 0:
                idraw += 1
                while len(raw[idraw]) < len(w):
                    idraw += 1
                word = w
            else:
                word = word + w.lstrip('##')
                while len(raw[idraw]) < len(word):
                    idraw += 1
            indexes[idraw].append(idx)

        flags = [len(idx) > 0 for idx in indexes]
        return indexes

    if pretrained.count('bert'):
        lengths = [len(sent) for sent in text]
        text = [' '.join(sent) for sent in text]
        if sep:
            text = [sent.split(' ' + Constants.SEP_WORD + ' ') for sent in text]
            tokens = [vocab.tokenizer.tokenize(sent[0]) + [Constants.SEP_WORD] + vocab.tokenizer.tokenize(sent[1]) + [
                Constants.SEP_WORD] for sent in text]
            text = [sent[0] for sent in text]
            index_dict = [get_dict(sent, length, raw.lower(), separate=sep) for sent, length, raw in
                          zip(tokens, lengths, text)]
        else:
            tokens = [vocab.tokenizer.tokenize(sent) for sent in text]
            index_dict = [get_dict(sent, length, raw.lower()) for sent, length, raw in zip(tokens, lengths, text)]
    else:
        index_dict = None
        tokens = [lower_sent(sent) for sent in text] if lower else text

    indexes = [vocab.convertToIdx(sent) for sent in tokens]

    return indexes, tokens, index_dict


def graph_data(opt, final_indexes, bert_indexes=None):
    # ========== get data ==========#
    # train_file, valid_file = json_load(opt.train_graph), json_load(opt.valid_graph)
    train_file, valid_file = json_load(opt.train_graph), json_load(opt.valid_graph)
    # train_ans_file, valid_ans_file = json_load(opt.train_ans_graph), json_load(opt.valid_ans_graph)
    train_data = get_graph_data(train_file, final_indexes[0])
    # train_ans_data = get_graph_data(train_ans_file, final_indexes[0])
    valid_data = get_graph_data(valid_file, final_indexes[1])
    # valid_ans_data = get_graph_data(valid_ans_file, final_indexes[1])
    # new_train_data = merge_graph(train_data, train_ans_data)
    # new_valid_data = merge_graph(valid_data, valid_ans_data)


    # print('valid_data',valid_data)

    # ========== build vocabulary ==========#

    print('build edge vocabularies')
    options = {'lower': True, 'mode': 'size', 'tgt': False,
               'size': opt.feat_vocab_size, 'frequency': opt.feat_words_min_frequency}


    edges_corpus = [edge_set for edges in train_data['matrix_undir'] + valid_data['matrix_undir'] for edge_set in edges]
    edges_binary_mask_corpus = [edge_set for edges in train_data['binary_edge_mask'] + valid_data['binary_edge_mask'] for edge_set in edges]

    # print('111111',edges_corpus)
    # assert 0==1

    edges_vocab = Vocab.from_opt(corpus=edges_corpus, opt=options, unk_need=False)
    edges_binary_mask_vocab=Vocab.from_opt(corpus=edges_binary_mask_corpus, opt=options, unk_need=False)
    train_edges_idxs = [convert_word_to_idx(sample, edges_vocab, lower=False)[0] for sample in train_data['matrix_undir']]
    print('train_edges_idxs',train_edges_idxs)

    train_edges_binary_mask_idxs=[convert_word_to_idx(sample, edges_binary_mask_vocab, lower=False)[0] for sample in train_data['binary_edge_mask']]
    valid_edges_idxs = [convert_word_to_idx(sample, edges_vocab, lower=False)[0] for sample in valid_data['matrix_undir']]
    # print('valid_edges_idxs', valid_edges_idxs[:1])

    valid_edges_binary_mask_idxs = [convert_word_to_idx(sample, edges_binary_mask_vocab, lower=False)[0] for sample in valid_data['binary_edge_mask']]

        # print('merged_tensor',merged_tensor)

    # ========== save data ===========#
    data = {'settings': opt,
            'dict': {
                     'edge': {
                         'matrix_undir': edges_vocab,
                         'binary_edge_mask': edges_binary_mask_vocab
                     }
                     },
            'train': {

                      'edge': {
                          'matrix_undir': train_edges_idxs,
                          'binary_edge_mask':train_edges_binary_mask_idxs


                      }
                      },
            'valid': {
                      'edge': {
                          'matrix_undir': valid_edges_idxs,
                          'binary_edge_mask':valid_edges_binary_mask_idxs
                      }
                      }
            }
    # print('data',data)
    # assert 0==1

    return data

def get_graph_data(raw, indexes):
    dataset = {'matrix_undir': [],  'binary_edge_mask': []}
    # print('len(raw)',len(raw))
    # print('indexes',indexes)
    for index in tqdm(indexes, desc=' - (Loading graph data) - '):
        sample = raw[index]
        matrix_undir,binary_edge_mask=sample['matrix_undir'],sample['binary_edge_mask']
        # print('nodes',nodes)
        # print('matrix_undir.shape',matrix_undir.shape)
        nodes_num = len( matrix_undir)
        # print('nodes_num',nodes_num)

        for idx in range(nodes_num):
            for i in range(nodes_num):
                if matrix_undir[idx][i] == '':
                    matrix_undir[idx][i] = Constants.PAD_WORD
                else:
                    continue
        dataset['matrix_undir'].append(matrix_undir)
        dataset['binary_edge_mask'].append(binary_edge_mask)

    return dataset
    # yield dataset

def main(opt):
    sequences, final_indexes, bert_indexes = sequence_data(opt)
    graphs = graph_data(opt, final_indexes, bert_indexes)
    print("Saving data ......")
    torch.save(sequences, opt.save_sequence_data)
    torch.save(graphs, opt.save_graph_data)
    print('Saving Datasets ......')
    print(' opt.batch_size', opt.batch_size)
    trainData = Dataset(sequences['train'], graphs['train'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=False)
    validData = Dataset(sequences['valid'], graphs['valid'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=False)
    torch.save(trainData, opt.train_dataset)
    torch.save(validData, opt.valid_dataset)
    # print(f"valid text shape:{sequences['valid'][].shape}")
    # print(f"valid graph shape:{graphs['valid'][].shape}")
    # assert 0 == 1
    print("Done .")
    # # print(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess.py')
    pargs.add_options(parser)
    opt = parser.parse_args()
    main(opt)