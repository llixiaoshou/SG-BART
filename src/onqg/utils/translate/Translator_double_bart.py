import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from torch.autograd import Variable

from onqg.utils.translate.Beam import Beam
import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch

from nltk.translate import bleu_score

from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('../pre_model/bart-base')


def add(tgt_list):
    tgt = []
    for b in tgt_list:
        tgt += b
    return tgt


def get_tokens(indexes, data):
    src, tgt = data['src'], data['tgt']
    srcs = [src[i] for i in indexes]
    golds = [[[w for w in tgt[i] if w not in [Constants.BOS_WORD, Constants.EOS_WORD]]] for i in indexes]
    return srcs, golds


class Translator(object):
    def __init__(self, opt, vocab, tokens, src_vocab):
        self.opt = opt
        self.max_token_seq_len = opt.max_token_tgt_len
        self.tokens = tokens
        if opt.gpus:
            cuda.set_device(opt.gpus[0])
        self.device = torch.device('cuda' if opt.gpus else 'cpu')
        self.vocab = vocab

        self.W1 = nn.Linear(768, 512).cuda()
        self.W2 = nn.Linear(1536, 512).cuda()

        # self.args = args

    def translate_batch(self, model, inputs,classification):

        with torch.no_grad():
            ### ========== Prepare data ========== ###
            if len(self.opt.gpus) > 1:
                model = model.module
            # src_seq_mask = src_seq_mask.to(inputs['seq-encoder']["src_seq"].device)
            graph_undir, edge_matrix_undir = inputs['graph-encoder']['graph_undir'], inputs['graph-encoder'][
                'edge_matrix_undir']
            input_ids, attention_mask = inputs['seq-encoder']["src_seq"].to(graph_undir.device), inputs['seq-encoder'][
                "src_mask"].to(graph_undir.device)
            # labels = inputs['decoder']['tgt_seq']

            rst = model.generate(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'),
                                 graph_undir=graph_undir.to('cuda'),
                                 edge_matrix_undir=edge_matrix_undir.to('cuda'), classification=classification.to('cuda'))

        #rst = model.generate(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'),output_hidden_states=True, num_beams=self.opt.beam_size,        early_stopping=True)

        # print('rst',rst.shape)

        return rst

    def eval_batch(self, model, inputs, max_length, classification,gold, copy_gold=None, copy_switch=None, batchIdx=None,
                   src_seq_mask=None):

        # def get_preds(seq, is_copy_seq=None, copy_seq=None, src_words=None, attn=None):
        #     # print('+++++', self.vocab.size)
        #     print('seq',seq)
        #     pred = [idx for idx in seq if idx not in [Constants.PAD, Constants.EOS]]
        #     # print('pred++',pred)
        #     # print('copy_seq',copy_seq)
        #     # print('src_words',src_words)
        #     # print('self.vocab.size',self.vocab.size)
        #     for i, _ in enumerate(pred):
        #         if self.opt.copy and is_copy_seq[i].item():
        #             try:
        #                 pred[i] = src_words[copy_seq[i].item() - self.vocab.size]
        #                 # print('pred[i]',pred[i])
        #             except:
        #                 print('len(copy_seq)', len(copy_seq))
        #             #     print('i', i)
        #         else:
        #             pred[i] = self.vocab.idxToLabel[pred[i]]
        #             # print('pred[i]2', pred[i])
        #     print('pred',pred)
        #     return pred

        def src_tokens(src_words):
            if self.opt.pretrained.count('bert'):
                tmp_word, tmp_idx = '', 0
                for i, w in enumerate(src_words):
                    if not w.startswith('##'):
                        if tmp_word:
                            src_words[tmp_idx] = tmp_word
                            for j in range(tmp_idx + 1, i):
                                src_words[j] = ''
                            tmp_word, tmp_idx = w, i
                        else:
                            tmp_word += w.lstrip('##')
                src_words[tmp_idx] = tmp_word
                for j in range(tmp_idx + 1, i):
                    src_words[j] = ''
            raw_words = [word for word in src_words if word != '']
            return src_words, raw_words

        golds, preds, paras = [], [], []

        all_hyp = self.translate_batch(model, inputs,classification)
        # print('all_hyp',all_hyp)

        # src_sents, golds = get_tokens(batchIdx, self.tokens)
        # print(batchIdx)
        # print(inputs)
        # assert 0 == 1
        golds_tokens = tokenizer.batch_decode(inputs['decoder']['tgt_seq'], skip_special_tokens=True)
        for item in golds_tokens:
            tmp = list(item.split(" "))
            golds.append([tmp])
        # print("golds tokens", golds_tokens)
        # print('golds',golds)
        # assert 0==1

        src_sents = tokenizer.batch_decode(inputs['seq-encoder']['src_seq'], skip_special_tokens=True)
        for item in src_sents:
            tmp = list(item.split(" "))
            paras.append(tmp)
            # print("golds tokens", golds_tokens)
        # print('paras', paras)
        # assert 0==1

        # print('batchIdx',batchIdx)
        for i, seq in tqdm(enumerate(all_hyp), mininterval=2, desc=' - (Translating)', leave=False):
            # print('seq',seq)
            m = tokenizer.decode(seq, skip_special_tokens=True)
            # print('m+++++++++++++++', m)
            n = list(m.split(" "))
            # print(n)
            # assert 0==1
            # seq = seqs[0]
            # m=src_sents[i]
            # print('m++',m)

            # src_words, raw_words = src_tokens(src_sents[i])
            # raw_words = [word for word in src_sents[i] if word != '']
            # print('src_words', src_words)
            # print('raw_words',raw_words)
            # if self.opt.copy:
            #     is_copy_seq, copy_seq = all_is_copy[i][0], all_copy_hyp[i][0]
            #     preds.append(get_preds(seq, is_copy_seq=is_copy_seq, copy_seq=copy_seq, src_words=src_words))
            # else:
            # preds.append(get_preds(seq))
            # preds.append(tokenizer.decode(seq, skip_special_tokens=True))
            # m=tokenizer.decode(seq, skip_special_tokens=True)
            # print('m+++++++++++++++',m)
            # paras.append(raw_words)
            preds.append(n)

        # print('paras',paras)
        # assert 0==1
        # print('preds',preds)
        # assert 0==1
        return {'gold': golds, 'pred': preds, 'para': paras}

    def eval_all(self, model, validData, output_sent=False):
        all_golds, all_preds = [], []
        if output_sent:
            all_paras = []

        for idx in tqdm(range(len(validData)), mininterval=2, desc='   - (Translating)   ', leave=False):
            ### ========== Prepare data ========== ###
            batch = validData[idx]
            # print('batch',batch)
            # assert 0==1
            # inputs, gold = preprocess_batch(batch, None, sparse=self.opt.sparse, feature=self.opt.feature,
            #                                 dec_feature=self.opt.dec_feature, copy=self.opt.copy,
            #                                 node_feature=self.opt.node_feature,
            #                                 device=self.device)

            inputs, gold,classification= preprocess_batch(batch, None, sparse=self.opt.sparse, feature=self.opt.feature,
                           dec_feature=self.opt.dec_feature, copy=self.opt.copy, node_feature=self.opt.node_feature,
                                                                          device=self.device)

            max_tgt_seq_length = inputs['max_tgt_seq_length']
            # copy_gold, copy_switch = copy[0], copy[1]
            ### ========== Translate ========== ###
            #            classification=None
            rst = self.eval_batch(model, inputs, max_tgt_seq_length,classification,gold,
                                  # copy_gold=copy_gold, copy_switch=copy_switch,
                                  batchIdx=batch['raw-index'])
            all_golds += rst['gold']
            all_preds += rst['pred']
            if output_sent:
                all_paras += rst['para']
        # print('all_golds',all_golds)
        # print('all_preds',all_preds)
        # assert 0==1
        bleu = bleu_score.corpus_bleu(all_golds, all_preds)
        print(bleu)
        if output_sent:
            return bleu, (all_golds, all_preds, all_paras)
        # print(bleu)
        # assert 0==1
        return bleu