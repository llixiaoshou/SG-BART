import numpy as np
import pickle
import spacy
import networkx as nx
import  re
import argparse
from transformers import BartTokenizer
from convert_span import test
import json
import codecs
Bart_tokenizer =BartTokenizer.from_pretrained('pre_model/bart-large')
token_nize_for_tokenize = spacy.load('en_core_web_trf')
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    # print('len(data)',len(data))
    data = data.split('\n')
    # print('len(data)',len(data))
    data = [sent.replace("\u200e", "").split(' ') for sent in data]
    # data = [sent.split(' ') for sent in data]
    # new_data = []
    # for item in data:
    #     item = item.replace("\u200e", "") # LRM
    #
    #     new_item = item.split(' ')
    #     new_data.append(new_item)
    # print('len(data)',len(data))
    return  data

def filter_data(data_list, opt):
    # print('data_list',data_list)
    # assert 0==1
    data_num = len(data_list)
    idx = list(range(len(data_list[0])))
    rst = [[] for _ in range(data_num)]
    final_indexes = []
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        src_len = len(src)
        # print('src_len',src_len)

        if src_len <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length:
            if len(src) * len(tgt) > 0 and src_len >= 5:  # TODO: fix this magic number
                rst[0].append(src)
                # rst[1].append([Constants.BOS_WORD] + tgt + [Constants.EOS_WORD])
                rst[1].append(tgt )

                final_indexes.append(i)
                for j in range(2, data_num):
                    sent = data_list[j][i]
                    rst[j].append(sent)

    print("change data size from " + str(len(idx)) + " to " + str(len(rst[0])))
    # print('final_indexes',final_indexes)
    return rst, final_indexes

def get_data(files, opt):
    # print('files',files)
    # assert 0==1
    src, tgt = load_file(files['src'])[0:12], load_file(files['tgt'])[0:12]
    # print('src',src)
    # assert 0==1
    # print('tgt',tgt)
    # src, tgt = load_file(files['src']), load_file(files['tgt'])
    data_list = [src, tgt]
    if opt.answer:
        data_list.append(load_file(files['ans'])[0:12])
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


def sequence_data(opt):
    # ========== get data ==========#
    train_files = {'src': opt.train_src, 'tgt': opt.train_tgt}
    valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt}

    train_data, train_final_indexes = get_data(train_files, opt)
    valid_data, valid_final_indexes = get_data(valid_files, opt)
    train_src, train_tgt = train_data['src'], train_data['tgt']
    valid_src, valid_tgt = valid_data['src'], valid_data['tgt']
    # print(' train_src', len(train_src))
    # assert 0==1

    return train_src ,valid_src


def dependency_adj_matrix(text):
    # print('text',text)
    # print('len(text)',len(text))
    # assert 0==1
    text = [w for w in text if w != ""]
    # print('text', text)
    m=" ".join(text)
    # print(m)
    document = token_nize_for_tokenize(m)
    n=[]
    for token in document:
        n.append(token.text)
    # print('n',n)
    # assert 0 == 1
    token_new=n
    n = " ".join(n)
    # print('len(n)'.len(n))
    words_pieces = Bart_tokenizer(n, add_special_tokens=False)
    # words_pieces = Bart_tokenizer(n, add_special_tokens=False, is_split_into_words=True)
    words_pieces_ids = words_pieces["input_ids"]
    words_pieces_tokens = Bart_tokenizer.convert_ids_to_tokens(words_pieces_ids)
    # print('words_pieces_tokens',words_pieces_tokens)
    # assert 0==1
    # print('words_pieces_tokens', words_pieces_ids)

    seq_len = len(words_pieces_tokens)
    # print(' seq_len', seq_len)
    # assert 0==1
    pos_sequence = []
    three_list = []
    star = 0
    single_token_list = []
    for index, token in enumerate(token_new):
        if index==0:
            acc=token
        else:
            acc=' '+token
        token_wordpice = Bart_tokenizer(acc, add_special_tokens=False)
        w_len = len(token_wordpice["input_ids"])
        end = star + w_len
        three_list.append([token, index, [star, end - 1]])
        star = end
        single_token_list += token_wordpice["input_ids"]
    # print("single token", single_token_list)
    # assert 0 == 1
    matrix_undir = [["" for _ in range(seq_len)] for __ in range(seq_len)]
    final_matrix_undir = [["" for _ in range(seq_len + 2)] for __ in range(seq_len + 2)]

    distance_list = []
    # print(three_list)
    pos_sequence.append("<unk>")
    sen_head = []
    # print('here', len(document))
    # assert 0==1
    for index, token in enumerate(document):
        # print(token.text)
        dep = token.dep_
        pos = token.pos_
        if token.head.i == token.i:
            sen_head.append(0)
        else:
            sen_head.append(token.head.i + 1)
        # print(token)
        # print(token,token.pos_)
        token_word_piece_list = []
        if (token.head.i < len(three_list) == True) & ((token.i) < len(three_list) == True):
            token_index = three_list[token.i][2]
            token_head_index = three_list[token.head.i][2]
        else:
            if token.i >= len(three_list):
                token_index = three_list[len(three_list) - 1][2]
                if token.head.i >= len(three_list):
                    token_index = three_list[len(three_list) - 1][2]
                else:
                    token_head_index = three_list[token.head.i][2]
            else:
                token_index = three_list[token.i][2]
                if token.head.i >= len(three_list):
                    token_head_index = three_list[len(three_list) - 1][2]
                else:
                    token_head_index = three_list[token.head.i][2]

        # print('token_index', token_index)
        # print('token_head_index', token_head_index)
        # print('dep',dep)
        # print('pos',pos)
        # print('sen_head',sen_head)

        token_l, token_r = token_index
        # print(token_l, token_r)
        head_l, head_r = token_head_index
        # print(head_l, head_r)
        # assert 0==1
        for i in range(token_l, token_r + 1):
            for j in range(head_l, head_r + 1):
                matrix_undir[i][j] = dep
                matrix_undir[j][i] = dep
                # cls +1

    for i in range(len(matrix_undir)):
        matrix_undir[i][i] = "SELF"

    matrix_undir = np.array(matrix_undir)
    # print('matrix_undir',matrix_undir)
    # assert 0==1
    padded_matrix = np.pad(matrix_undir, ((1, 1), (0, 0)), mode='constant', constant_values="")
    # print('padded_matrix',padded_matrix)
    final_matrix_undir = np.pad(padded_matrix, ((0, 0), (1, 1)), mode='constant', constant_values="")
    # print('final_matrix_undir',final_matrix_undir)
    # assert 0==1
    # rows, cols = matrix_undir.shape
    # final_matrix_undir=np.empty((rows + 2, cols + 2), dtype=matrix_undir.dtype)
    # final_matrix_undir[:-2, :-2] = matrix_undir
    # final_matrix_undir[-2:, :] = ''
    # final_matrix_undir[:, -2:] = ''


    # assert 0==1
    # # padded_matrix = np.pad(matrix_undir, ((1, 1), (0, 0)), mode='constant', constant_values="")
    # padded_matrix = np.pad(matrix_undir, ((0, 0), (1, 1)), mode='constant', constant_values="")
    # print('padded_matrix',padded_matrix)
    # # assert 0==1
    # final_matrix_undir = np.pad(padded_matrix, ((0, 0), (1, 1)), mode='constant', constant_values="")
    # print('final_matrix_undir',final_matrix_undir)
    # assert 0==1

    # print(token_new)
    # assert 0 == 1
    tokens, input_span_mask, record_mask=test(token_new,[sen_head],Bart_tokenizer)
    # print('input_span_mask',input_span_mask)
    # assert 0==1

    padded_matrix = np.pad(input_span_mask, ((1, 1), (0, 0)), mode='constant', constant_values='[PAD]')
    # print('padded_matrix', padded_matrix)
    final_binary_mask = np.pad(padded_matrix, ((0, 0), (1, 1)), mode='constant', constant_values='[PAD]')
    # print('final_binary_mask', final_binary_mask)
    # assert 0 == 1



    # rows, cols = input_span_mask.shape
    # final_binary_mask= np.empty((rows + 2, cols + 2), dtype=input_span_mask.dtype)
    # final_binary_mask[:-2, :-2] = input_span_mask
    # final_binary_mask[-2:, :] = '[PAD]'
    # final_binary_mask[:, -2:] = '[PAD]'




    # assert 0==1
    # padded_matrix = np.pad( input_span_mask, ((1, 1), (0, 0)), mode='constant', constant_values="[PAD]")
    # final_binary_mask=np.pad(padded_matrix, ((0, 0), (1, 1)), mode='constant', constant_values="[PAD]")

    # print('final_binary_mask',final_binary_mask)
    # assert 0==1
    return final_matrix_undir,final_binary_mask

# def process(text_src,text_ans):
#
#     # print('len(texts)',len(texts))
#     # assert 0==1
#     final_graphs=[]
#         # {'matrix_undir':[],'binary_mask':[]}
#     for i in range(0,len(text_src)):
#         data = {}
#         text=text_src[i]
#         final_text_matrix_undir, final_text_binary_mask = dependency_adj_matrix(text)
#         # print('final_text_matrix_undir',final_text_matrix_undir)
#         row_text=len(final_text_matrix_undir)
#         # print('row_text',row_text)
#         col_text=len(final_text_matrix_undir)
#         # print('col_text',col_text)
#         # assert 0==1
#         # a=len(final_text_matrix_undir)
#         # print('a',a)
#         # data['matrix_undir'] = final_matrix_undir.tolist()
#         # data['binary_edge_mask'] = final_binary_mask.tolist()
#         ans=text_ans[i]
#         final_ans_matrix_undir, final_ans_binary_mask = dependency_adj_matrix(ans)
#         b=final_ans_matrix_undir
#         # print('b',b)
#         len_text=len(final_text_matrix_undir)+len(final_ans_matrix_undir)
#         final_matrix_undir = [["" for _ in range(len_text-1)] for __ in range(len_text-1)]
#         final_binary_mask=[["[PAD]" for _ in range(len_text-1)] for __ in range(len_text-1)]
#
#         final_matrix_undir=np.array(final_matrix_undir,dtype='<U10')
#         # print('final_matrix_undir',final_matrix_undir)
#
#         final_binary_mask = np.array(final_binary_mask, dtype='<U5')
#
#
#
#         # assert 0==1
#         final_matrix_undir[0: row_text,0:col_text]=final_text_matrix_undir
#         # print('final_matrix_undir',final_matrix_undir)
#         final_matrix_undir[row_text:, col_text:] = final_ans_matrix_undir[1:,1:]
#         # print('final_matrix_undir', final_matrix_undir)
#
#         final_binary_mask[0:row_text, 0:col_text] = final_text_binary_mask
#         final_binary_mask[row_text:, col_text:] = final_ans_binary_mask[1:, 1:]
#
#         # print('final_binary_mask',final_binary_mask)
#         # assert 0==1
#         # print(' final_ans_matrix_undir', final_ans_matrix_undir)
#         # print('final_ans_binary_mask',final_ans_binary_mask)
#         data['matrix_undir'] = final_matrix_undir.tolist()
#         # a=final_ans_matrix_undir.tolist()
#         data['binary_edge_mask'] = final_binary_mask.tolist()
#
#         final_graphs.append(data)
#
#
#     # print('final_graphs',final_graphs)
#     # print(len(final_graphs["binary_mask"]))
#
#     return final_graphs
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-train_src', default='dataset/text-data/train.src.txt', type=str)
#     parser.add_argument('-valid_src', default='dataset/text-data/valid.src.txt', type=str)
#     parser.add_argument('-train_tgt', default='dataset/text-data/train.tgt.txt', type=str)
#     parser.add_argument('-valid_tgt', default='dataset/text-data/valid.tgt.txt', type=str)
#     parser.add_argument('-train_ans', default='dataset/text-data/train.ans.txt', type=str)
#     parser.add_argument('-valid_ans', default='dataset/text-data/valid.ans.txt', type=str)
#     parser.add_argument('-train_edges', default='dataset/json-data/train_edges.json')
#     parser.add_argument('-valid_edges', default='dataset/json-data/valid_edges.json')
#     parser.add_argument('-answer', default=True, action='store_true')
#     parser.add_argument('-feature', default=False, action='store_true', help="whether to incorporate word feature information")
#     parser.add_argument('-src_seq_length', default=200)
#     parser.add_argument('-tgt_seq_length', default=50)
#
#     opt = parser.parse_args()
#     train_files = {'src': opt.train_src, 'tgt': opt.train_tgt,'ans':opt.train_ans}
#     # valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt}
#     train_data, train_final_indexes = get_data(train_files, opt)
#
#     train_src = train_data['src']
#     train_ans=train_data['ans']
#
#     valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt,'ans':opt.valid_ans}
#     valid_data, valid_final_indexes = get_data(valid_files, opt)
#     # print('valid_data',valid_data)
#     valid_src = valid_data['src']
#
#     # train_src,valid_src = sequence_data(opt)
#     # print('train_src',train_src[1])
#     # train_src=[["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?"]]
#     # print('train_src',train_src)
#     train_final_graps=process(train_src,train_ans)
#
#     valid_final_graps=process(valid_src,train_ans)
#
#
#     json_dump(train_final_graps, opt.train_edges)
#     json_dump(valid_final_graps, opt.valid_edges)
def process(texts):
    # print('len(texts)',len(texts))
    # assert 0==1
    final_graphs=[]
        # {'matrix_undir':[],'binary_mask':[]}
    for i in range(0,len(texts)):
        data = {}
        text=texts[i]
        final_matrix_undir, final_binary_mask = dependency_adj_matrix(text)
        data['matrix_undir'] = final_matrix_undir.tolist()
        data['binary_edge_mask'] = final_binary_mask.tolist()
        final_graphs.append(data)


    # print('final_graphs',final_graphs)
    # print(len(final_graphs["binary_mask"]))

    return final_graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='dataset/text-data/train.src.txt', type=str)
    parser.add_argument('-valid_src', default='dataset/text-data/valid.src.txt', type=str)
    parser.add_argument('-train_tgt', default='dataset/text-data/train.tgt.txt', type=str)
    parser.add_argument('-valid_tgt', default='dataset/text-data/valid.tgt.txt', type=str)
    parser.add_argument('-train_ans', default='dataset/text-data/train.ans.txt', type=str)
    parser.add_argument('-valid_ans', default='dataset/text-data/valid.ans.txt', type=str)
    parser.add_argument('-train_edges', default='dataset/json-data/train_edges.json')
    parser.add_argument('-valid_edges', default='dataset/json-data/valid_edges.json')
    parser.add_argument('-answer', default=False, action='store_true')
    parser.add_argument('-feature', default=False, action='store_true', help="whether to incorporate word feature information")
    parser.add_argument('-src_seq_length', default=200)
    parser.add_argument('-tgt_seq_length', default=50)


    opt = parser.parse_args()

    train_files = {'src': opt.train_src, 'tgt': opt.train_tgt}
    # valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt}
    train_data, train_final_indexes = get_data(train_files, opt)
    train_src = train_data['src']

    valid_files = {'src': opt.valid_src, 'tgt': opt.valid_tgt}
    valid_data, valid_final_indexes = get_data(valid_files, opt)
    valid_src = valid_data['src']

    # train_src,valid_src = sequence_data(opt)
    # print('train_src',train_src[1])
    # train_src=[["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?"]]

    train_final_graps=process(train_src)
    valid_final_graps=process(valid_src)


    json_dump(train_final_graps, opt.train_edges)
    json_dump(valid_final_graps, opt.valid_edges)











