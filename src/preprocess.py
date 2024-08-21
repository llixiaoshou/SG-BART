import math
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

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))


def load_vocab(filename):
    vocab_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    text = [word.split(' ') for word in text]
    vocab_dict = {word[0]: word[1:] for word in text}
    vocab_dict = {k: [float(d) for d in v] for k, v in vocab_dict.items()}
    return vocab_dict


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    data = data.split('\n')
    data = [sent.split(' ') for sent in data]
    return data


def filter_data(data_list, opt):
    data_num = len(data_list)
    idx = list(range(len(data_list[0])))
    rst = [[] for _ in range(data_num)]

    final_indexes = []
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        src_len = len(src)
        # print('src_len',src_len)
        if src_len <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length:
            if len(src) * len(tgt) > 0 and src_len >= 10:  # TODO: fix this magic number
                rst[0].append(src)
                # rst[1].append([Constants.BOS_WORD] + tgt + [Constants.EOS_WORD])
                rst[1].append(tgt )
                final_indexes.append(i)
                for j in range(2, data_num):
                    sent = data_list[j][i]
                    rst[j].append(sent)

    print("change data size from " + str(len(idx)) + " to " + str(len(rst[0])))
    return rst, final_indexes


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


def convert_word_to_idx2(data):
    """
     转 01
    """
    for d in data:
        for i in d:
            for index in range(i.shape[0]):
                if i[index] != 0:
                    i[index] = 1
    return data

    # return new_data


def get_embedding(vocab_dict, vocab):
    def get_vector(idx):
        word = vocab.idxToLabel[idx]
        if idx in vocab.special or word not in vocab_dict:
            vector = torch.tensor([])
            vector = vector.new_full((opt.word_vec_size,), 1.0)
            vector.normal_(0, math.sqrt(6 / (1 + vector.size(0))))
        else:
            vector = torch.Tensor(vocab_dict[word])
        return vector

    embedding = [get_vector(idx) for idx in range(vocab.size)]
    embedding = torch.stack(embedding)
    # print(embedding.size())
    return embedding


def get_data(files, opt):
    # print('files',files)
    # assert 0==1
    src, tgt = load_file(files['src'])[0:1], load_file(files['tgt'])[0:1]
    # print('src',src)
    # print('tgt',tgt)
    # src, tgt = load_file(files['src']), load_file(files['tgt'])
    data_list = [src, tgt]
    if opt.answer:
        data_list.append(load_file(files['ans']))
    if opt.feature:
        data_list +=[load_file(filename) for filename in files['feats']]
    data_list, final_indexes = filter_data(data_list, opt)

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
    # print('rst',rst)
    return rst, final_indexes


def merge_ans(src, ans):
    rst = [s + [Constants.SEP_WORD] + a + [Constants.SEP_WORD] for s, a in zip(src, ans)]
    return rst


def wrap_copy_idx(splited, tgt, tgt_vocab, bert, vocab_dict):
    def map_src(sp):
        sp_split = {}
        if bert:
            tmp_idx = 0
            tmp_word = ''
            for i, w in enumerate(sp):
                if not w.startswith('##'):
                    if tmp_word:
                        sp_split[tmp_word] = tmp_idx
                    tmp_word = w
                    tmp_idx = i
                else:
                    tmp_word += w.lstrip('##')
            sp_split[tmp_word] = tmp_idx
        else:
            sp_split = {w: idx for idx, w in enumerate(sp)}
        return sp_split

    def wrap_sent(sp, t):
        sp_dict = map_src(sp)
        swt, cpt = [0 for w in t], [0 for w in t]
        for i, w in enumerate(t):
            # if w not in tgt_vocab.labelToIdx or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
            if w not in tgt_vocab.labelToIdx or w not in vocab_dict or tgt_vocab.frequencies[
                tgt_vocab.labelToIdx[w]] <= 1:
                if w in sp_dict:
                    swt[i] = 1
                    cpt[i] = sp_dict[w]
        return torch.Tensor(swt), torch.LongTensor(cpt)

    copy = [wrap_sent(sp, t) for sp, t in zip(splited, tgt)]
    switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
    return [switch, cp_tgt]


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
    valid_src, valid_tgt = valid_data['src'], valid_data['tgt']

    train_ans = train_data['ans'] if opt.answer else None
    # print(' train_ans', train_ans)
    valid_ans = valid_data['ans'] if opt.answer else None
    train_feats = train_data['feats'] if opt.feature else None
    valid_feats = valid_data['feats'] if opt.feature else None

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # tokenizer = AutoTokenizer.from_pretrained('../pre_model/bart-base')

    tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-base')

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

    # print('train_src',train_src[0])

    a=tokenizer.tokenize(' '.join(train_src[0]))
    # print('a',a)
    # assert 0==1
    a=[]

    for train_tokens_input in train_src:
        # print('train_tokens_input',train_tokens_input)
        # print('seq_src',len(train_tokens_input))
        # assert 0==1
        for item in train_tokens_input:
            # print('item',item)
            item.lower().strip()
            # print('item',item)
            a.append(item)
            # assert 0==1

        # print('a',a)
        # assert 0==1
        # c=train_tokens_input.strip()
        # print('c',c)
        # assert 0==1
        # # a=tokenizer.tokenize(train_tokens_input)
        # # # print('a',a)
        # # assert 0==1
        # wordpieces = tokenizer(train_tokens_input)
        #
        # print('wordpieces',wordpieces)
        # a=wordpieces['input_ids']
        # print('a',a)
        #
        #
        # for item in a:
        #     print('item',item)
        #     b =str(tokenizer.decode(item))
        #     print('b',b)
        #
        # a= tokenizer.decode(wordpieces)
        # print('a',a)
        # # assert 0==1
        # # print('train_tokens_input',train_tokens_input)

        train_tokens = tokenizer(" ".join(train_tokens_input), return_tensors='pt', max_length=200, padding="max_length", truncation=True)
        input_id, attention_mask = train_tokens["input_ids"][0], train_tokens['attention_mask'][0]
        # print('attention_mask',attention_mask)
        # print('attention_mask.shape',attention_mask.shape)
        src_length =torch.sum(attention_mask==1)
        # print('src_length', src_length)
        # assert 0==1

        max_src_length = torch.max(src_length)
        # print('src_seq.shape',src_seq.shape)

        # print('max_src_length',max_src_length)

        # assert 0==1
        # print('input_id',input_id)
        # print('input_id.shape',input_id.shape)
        # assert 0==1
        #
        # assert 0==1
        # print('attention_mask',attention_mask)
        train_src_input_ids.append(input_id)
        train_src_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_src_tokens.append(temp)


    valid_src_input_ids, valid_src_attention_masks, valid_src_tokens = [], [], []
    for valid_tokens_input in valid_src:
        valid_tokens = tokenizer(" ".join(valid_tokens_input), return_tensors='pt', max_length=200, padding="max_length",
                                        truncation=True)
        input_id, attention_mask = valid_tokens["input_ids"][0], valid_tokens['attention_mask'][0]
        valid_src_input_ids.append(input_id)
        valid_src_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_src_tokens.append(temp)


    train_tgt_input_ids, train_tgt_attention_masks,train_tgt_tokens = [],[],[]
    for train_tokens_input in train_tgt:
        train_tgt_token = tokenizer(" ".join(train_tokens_input), return_tensors='pt', max_length=200, padding="max_length",
                                        truncation=True)
        input_id, attention_mask = train_tgt_token["input_ids"][0], train_tgt_token['attention_mask'][0]
        train_tgt_input_ids.append(input_id)
        train_tgt_attention_masks.append(attention_mask)

        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_tgt_tokens.append(temp)

    valid_tgt_input_ids, valid_tgt_attention_masks,valid_tgt_tokens = [], [], []
    for valid_tokens_input in valid_tgt:
        valid_tgt_token = tokenizer(" ".join(valid_tokens_input), return_tensors='pt', max_length=200, padding="max_length",
                                        truncation=True)
        input_id, attention_mask = valid_tgt_token["input_ids"][0], valid_tgt_token['attention_mask'][0]
        valid_tgt_input_ids.append(input_id)
        valid_tgt_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_tgt_tokens.append(temp)

    # ans 分词（BART分词）
    train_ans_input_ids, train_ans_attention_masks, train_ans_tokens = [], [], []
    for train_tokens_input in train_ans:
        train_tokens = tokenizer(" ".join(train_tokens_input), return_tensors='pt', max_length=200,
                                 padding="max_length",truncation=True)
        input_id, attention_mask = train_tokens["input_ids"][0], train_tokens['attention_mask'][0]
        train_ans_input_ids.append(input_id)
        train_ans_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        train_ans_tokens.append(temp)
    valid_ans_input_ids, valid_ans_attention_masks, valid_ans_tokens = [], [], []
    for valid_tokens_input in valid_ans:
        valid_tokens = tokenizer(" ".join(valid_tokens_input), return_tensors='pt', max_length=200,
                                 padding="max_length",
                                 truncation=True)
        input_id, attention_mask = valid_tokens["input_ids"][0], valid_tokens['attention_mask'][0]
        valid_ans_input_ids.append(input_id)
        valid_ans_attention_masks.append(attention_mask)
        temp = []
        for i in input_id:
            temp.append(tokenizer.decode(i))
        valid_ans_tokens.append(temp)



    train_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                       in zip(train_feats, feats_vocab)] if opt.feature else None
    valid_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                       in zip(valid_feats, feats_vocab)] if opt.feature else None


    data = {'settings': opt,
            'dict': {'src': src_vocab,
                     'tgt': tgt_vocab,
                     },
            'train': {'src': train_src_input_ids,
                      'tgt': train_tgt_input_ids,
                      'ans': train_ans_input_ids,
                      'feature': train_feat_idxs,
                      'src_mask':train_src_attention_masks,
                      'tgt_mask':train_tgt_attention_masks,
                      'ans_mask':train_ans_attention_masks,
                      },
            'valid': {'src': valid_src_input_ids,
                      'tgt': valid_tgt_input_ids,
                      'ans': valid_ans_input_ids,
                      'feature': valid_feat_idxs,
                      'src_mask': valid_src_attention_masks,
                      'tgt_mask': valid_tgt_attention_masks,
                      'ans_mask': valid_ans_attention_masks,
                      'tokens': {'src': valid_src_tokens,
                                 'tgt': valid_tgt_tokens}
                      }
            }

    return data, (train_final_indexes, valid_final_indexes), (None, None)


def get_graph_data(raw, indexes):
    def get_edges(edges):
        edge_in = [[dep for dep in edge_set] for edge_set in edges]
        edge_out = [[dep for dep in edge_set] for edge_set in edges]
        for idx, edge_set in enumerate(zip(edge_in, edge_out)):
            set_in, set_out = edge_set[0], edge_set[1]
            for i, deps in enumerate(zip(set_in, set_out)):
                dep_in, dep_out = deps[0], deps[1]
                if dep_in not in ['self', 'similar', Constants.PAD_WORD] and not dep_in.startswith('re-'):
                    edge_in[idx][i] = Constants.PAD_WORD
                if dep_out not in ['self', 'similar', Constants.PAD_WORD] and dep_out.startswith('re-'):
                    edge_out[idx][i] = Constants.PAD_WORD
        return edge_in, edge_out

    # dataset = {'indexes': [], 'type': [], 'is_ans': [], 'pos': [], 'is_tgt': [], 'edges': [], 'edge_in': [],
    #            'edge_out': [], 'pos_edges': [], 'span_mask': [], 'edge_mask': []}
    dataset = {'indexes': [], 'type': [], 'is_ans': [], 'pos': [], 'is_tgt': [], 'edges': [], 'edge_in': [],
               'edge_out': [], 'pos_edges': [], 'span_mask': [], 'edge_mask': []}

    for index in tqdm(indexes, desc='   - (Loading graph data) -   '):
        sample = raw[index]
        nodes, edges, pos_edges, span_mask = sample['nodes'], sample['edges'], sample['pos_edges'], sample['mask_edges']
        # print('nodes',nodes)
        nodes_num = len(nodes)
        node_indexes, types, ans_tags, pos_tags, tgt_tags = [], [], [], [], []
        for idx, node in enumerate(nodes):
            node_indexes.append(node['index'])
            types.append(node['type'])
            ans_tags.append(node['ans'])
            pos_tags.append(node['pos'])
            tgt_tags.append(node['tag'])
            for i in range(nodes_num):
                if edges[idx][i] == '':
                    edges[idx][i] = Constants.PAD_WORD
                else:
                    edges[idx][i] = edges[idx][i].lower()
                    if edges[idx][i] not in ['similar', 'self'] and not edges[idx][i].startswith('re-'):
                        # edges[i][idx] = 1
                        edges[i][idx] = 're-' + edges[idx][i]

                if pos_edges[idx][i] == '':
                    pos_edges[idx][i] = Constants.PAD_WORD
                if span_mask[idx][i] == '':
                    span_mask[idx][i] = Constants.PAD_WORD

        dataset['indexes'].append(node_indexes)
        dataset['type'].append(types)
        dataset['is_ans'].append(ans_tags)
        dataset['pos'].append(pos_tags)
        dataset['is_tgt'].append(tgt_tags)
        dataset['edges'].append(edges)
        dataset['pos_edges'].append(pos_edges)
        dataset['span_mask'].append(span_mask)
        edge_in, edge_out = get_edges(edges)
        dataset['edge_in'].append(edge_in)
        dataset['edge_out'].append(edge_out)
    dataset['features'] = [dataset['type'], dataset['is_ans'], dataset['pos'], dataset['is_tgt']]

    # print('+++++++',dataset)
    return dataset


def process_bert_index(dicts, indexes):
    indexes = [dicts[idx] for idx in indexes]
    indexes = [i for idx in indexes for i in idx]

    return indexes


def graph_data(opt, final_indexes, bert_indexes=None):
    # ========== get data ==========#
    print(opt.train_graph)
    print(opt.valid_graph)
    # assert 0==1
    train_file, valid_file = json_load(opt.train_graph), json_load(opt.valid_graph)
    train_data = get_graph_data(train_file, final_indexes[0])
    # print('train_data',train_data)
    # assert 0==1
    valid_data = get_graph_data(valid_file, final_indexes[1])
    # print('valid_data',valid_data)

    # ========== build vocabulary ==========#
    print('build node feature vocabularies')
    options = {'lower': False, 'mode': 'size', 'tgt': False,
               'size': opt.feat_vocab_size, 'frequency': opt.feat_words_min_frequency}

    feats_vocab = [Vocab.from_opt(corpus=ft + fv, opt=options) for
                   ft, fv in zip(train_data['features'], valid_data['features'])] if opt.node_feature else None
    print('build edge vocabularies')
    options = {'lower': True, 'mode': 'size', 'tgt': False,
               'size': opt.feat_vocab_size, 'frequency': opt.feat_words_min_frequency}

    # print('测试', train_data['edges'])

    edges_corpus = [edge_set for edges in train_data['edges'] + valid_data['edges'] for edge_set in edges]
    print('111111',edges_corpus)
    assert 0==1


    edge_in_corpus = [edge_set for edges in train_data['edge_in'] + valid_data['edge_in'] for edge_set in edges]
    edge_out_corpus = [edge_set for edges in train_data['edge_out'] + valid_data['edge_out'] for edge_set in edges]
    pos_edges_corpus = [edge_set for edges in train_data['pos_edges'] + valid_data['pos_edges'] for edge_set in edges]
    edge_smask_corpus = [edge_set for edges in train_data['span_mask'] + valid_data['span_mask'] for edge_set in edges]

    edges_vocab = Vocab.from_opt(corpus=edges_corpus, opt=options, unk_need=False)
    edge_in_vocab = Vocab.from_opt(corpus=edge_in_corpus, opt=options, unk_need=False)
    edge_out_vocab = Vocab.from_opt(corpus=edge_out_corpus, opt=options, unk_need=False)
    pos_edges_vocab = Vocab.from_opt(corpus=pos_edges_corpus, opt=options, unk_need=False)
    edge_smask_vocab = Vocab.from_opt(corpus=edge_smask_corpus, opt=options, unk_need=False)

    # ========== word to index ==========#
    # print(len(train_data['features']))
    train_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                       in zip(train_data['features'], feats_vocab)] if opt.node_feature else None
    valid_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                       in zip(valid_data['features'], feats_vocab)] if opt.node_feature else None

    # print("测试",train_data['edges'])
    train_edges_idxs = [convert_word_to_idx(sample, edges_vocab, lower=False)[0] for sample in train_data['edges']]

    # print('测试++',train_edges_idxs)

    valid_edges_idxs = [convert_word_to_idx(sample, edges_vocab, lower=False)[0] for sample in valid_data['edges']]
    # print('测试',valid_edges_idxs)
    train_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in
                          train_data['edge_in']]
    valid_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in
                          valid_data['edge_in']]
    train_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in
                           train_data['edge_out']]
    valid_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in
                           valid_data['edge_out']]
    train_pos_edges_idxs = [convert_word_to_idx(sample, pos_edges_vocab, lower=False)[0] for sample in
                            train_data['pos_edges']]
    valid_pos_edges_idxs = [convert_word_to_idx(sample, pos_edges_vocab, lower=False)[0] for sample in
                            valid_data['pos_edges']]
    train_edge_smask_idxs = [convert_word_to_idx(sample, edge_smask_vocab, lower=False)[0] for sample in
                             train_data['span_mask']]
    valid_edge_smask_idxs = [convert_word_to_idx(sample, edge_smask_vocab, lower=False)[0] for sample in
                             valid_data['span_mask']]

    new_train_edges_idxs = copy.deepcopy(train_edges_idxs)
    new_valid_edges_idxs = copy.deepcopy(valid_edges_idxs)
    train_edge_masks = convert_word_to_idx2(new_train_edges_idxs)
    valid_edge_masks = convert_word_to_idx2(new_valid_edges_idxs)
    # print("2:", train_edges_idxs)

    # ========== process indexes ===========#
    if opt.pretrained:
        train_indexes, valid_indexes = bert_indexes[0], bert_indexes[1]
        train_data['indexes'] = [[process_bert_index(dicts, node) for node in sent] for sent, dicts in
                                 zip(train_data['indexes'], train_indexes)]
        valid_data['indexes'] = [[process_bert_index(dicts, node) for node in sent] for sent, dicts in
                                 zip(valid_data['indexes'], valid_indexes)]

    # ========== save data ===========#
    data = {'settings': opt,
            'dict': {'feature': feats_vocab,
                     'edge': {
                         'edges++': edges_vocab,
                         'in': edge_in_vocab,
                         'out': edge_out_vocab,
                         'pos_edges': pos_edges_vocab,
                         'span_mask': edge_smask_vocab,

                     }
                     },
            'train': {'index': train_data['indexes'],
                      'feature': train_feat_idxs,
                      'edge': {
                          'edges++': train_edges_idxs,
                          'in': train_edge_in_idxs,
                          'out': train_edge_out_idxs,
                          'pos_edges': train_pos_edges_idxs,
                          'span_mask': train_edge_smask_idxs,
                          'edge_mask': train_edge_masks
                      }
                      },
            'valid': {'index': valid_data['indexes'],
                      'feature': valid_feat_idxs,
                      'edge': {
                          'edges++': valid_edges_idxs,
                          'in': valid_edge_in_idxs,
                          'out': valid_edge_out_idxs,
                          'pos_edges': valid_pos_edges_idxs,
                          'span_mask': valid_edge_smask_idxs,
                          'edge_mask': valid_edge_masks
                      }
                      }
            }
    # print('data',data)
    # assert 0==1

    return data


def main(opt):
    sequences, final_indexes, bert_indexes = sequence_data(opt)

    graphs = graph_data(opt, final_indexes, bert_indexes)
    print("Saving data ......")
    torch.save(sequences, opt.save_sequence_data)
    #torch.save(graphs, opt.save_graph_data)
    print('Saving Datasets ......')

    trainData = Dataset(sequences['train'], graphs['train'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=False)

    validData = Dataset(sequences['valid'], graphs['valid'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=False)

    torch.save(trainData, opt.train_dataset)
    torch.save(validData, opt.valid_dataset)
    print("Done .")
    # print(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess.py')
    pargs.add_options(parser)
    opt = parser.parse_args()
    main(opt)
