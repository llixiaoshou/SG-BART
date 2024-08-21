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

tokenizer = BartTokenizer.from_pretrained("../pre_model/bart-base")

def add(tgt_list):
    tgt = []
    for b in tgt_list:
        tgt += b
    return tgt


def get_tokens(indexes, data):
    # print('indexes',indexes)
    src, tgt = data['src'], data['tgt']
    # print('src',src)
    # print('tgt',tgt)
    srcs = [src[i] for i in indexes]
    # print('srcs', srcs)
    golds = [[[w for w in tgt[i] if w not in ['<s>', '</s>','<pad>']]] for i in indexes]
    # print('golds',golds)
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


    def translate_batch(self, model, inputs, max_length):
        
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq1 = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq2 = torch.stack(dec_partial_seq1).to(self.device)
            dec_partial_seq = dec_partial_seq2.view(-1, len_dec_seq)
            return dec_partial_seq
        
        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos
        
        def collect_active_inst_idx_list(inst_beams, pred_prob, copy_pred_prob, inst_idx_to_position_map, n_bm):

            # print('inst_idx_to_position_map',inst_idx_to_position_map)
            active_inst_idx_list = []
            pred_prob = pred_prob.unsqueeze(0).view(len(inst_idx_to_position_map), n_bm, -1)
            # print('pred_prob',pred_prob)
            copy_pred_prob = None if not self.opt.copy else copy_pred_prob.unsqueeze(0).view(len(inst_idx_to_position_map), n_bm, -1)
            # print('copy_pred_prob',copy_pred_prob)
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                # print('inst_position',inst_position)
                copy_prob = None if not self.opt.copy else copy_pred_prob[inst_position]
                # print('copy_prob',copy_prob)
                is_inst_complete = inst_beams[inst_idx].advance(pred_prob[inst_position], copy_prob)
                # print('is_inst_complete',is_inst_complete)
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm, layer=False):
            ''' Collect tensor parts associated to active instances. '''
            tmp_beamed_tensor = beamed_tensor[0] if layer else beamed_tensor
            _, *d_hs = tmp_beamed_tensor.size()

            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor if layer else [beamed_tensor]

            beamed_tensor = [layer_b_tensor.view(n_prev_active_inst, -1) for layer_b_tensor in beamed_tensor]
            beamed_tensor = [layer_b_tensor.index_select(0, curr_active_inst_idx) for layer_b_tensor in beamed_tensor]
            beamed_tensor = [layer_b_tensor.view(*new_shape) for layer_b_tensor in beamed_tensor]
            
            beamed_tensor = beamed_tensor if layer else beamed_tensor[0]
            
            return beamed_tensor

        with torch.no_grad():
            ### ========== Prepare data ========== ###
            if len(self.opt.gpus) > 1:
                model = model.module
            # print('inputs',inputs)
            # print('+++++++',inputs['seq-encoder']["src_mask"].shape)
            src_seq_mask = inputs['seq-encoder']["src_mask"].cuda()
            # print('src_seq_mask',src_seq_mask.shape)
            # print('++++++++',inputs['seq-encoder']["src_seq"].shape)
            # print('src_seq_mask',src_seq_mask.shape)
            # rst = model.generate(inputs['seq-encoder']["src_seq"], attention_mask=src_seq_mask, output_hidden_states=True,
            #                  # graph_undir=inputs['graph-encoder']['graph_undir'],num_beams=self.opt.beam_size,early_stopping=True)

            # print('src_seq++',inputs['seq-encoder']["src_seq"].shape)
            # print('src_mask+++',inputs['seq-encoder']["src_mask"].cuda().shape)

            # rst = model.generate(inputs['seq-encoder']["src_seq"], attention_mask=src_seq_mask,
            #                      output_hidden_states=True, num_beams=self.opt.beam_size,
            #                      early_stopping=True)

            # print('inputs',inputs)
            rst = model.generate(inputs['seq-encoder']["src_seq"], attention_mask=src_seq_mask,
                                  num_beams=self.opt.beam_size)

        # print('rst',rst)
        return rst
    
    def eval_batch(self, model, inputs, max_length, gold, copy_gold=None, copy_switch=None, batchIdx=None):
        
        def get_preds(seq, is_copy_seq=None, copy_seq=None, src_words=None, attn=None):
            # print('+++++', self.vocab.size)
            # print('seq',seq)
            pred = [idx for idx in seq if idx not in [Constants.PAD, Constants.EOS]]
            # print('pred++',pred)
            # print('copy_seq',copy_seq)
            # print('src_words',src_words)
            # print('self.vocab.size',self.vocab.size)
            for i, _ in enumerate(pred):
                if self.opt.copy and is_copy_seq[i].item():
                    try:
                        pred[i] = src_words[copy_seq[i].item() - self.vocab.size]
                        # print('pred[i]',pred[i])
                    except:
                        print('len(copy_seq)', len(copy_seq))
                    #     print('i', i)
                else:
                    pred[i] = self.vocab.idxToLabel[pred[i]]
                    # print('pred[i]2', pred[i])
            # print('pred',pred)
            return pred
        
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
            
        golds, preds, paras,m = [], [], [],[]
        if self.opt.copy:
            all_hyp, _, all_is_copy, all_copy_hyp = self.translate_batch(model, inputs, max_length)
        else:
            all_hyp = self.translate_batch(model, inputs, max_length)
        # print('all_hyp',all_hyp)
        # print('batchIdx',batchIdx)
        # print('++++++',self.tokens)
        src_sents, golds = get_tokens(batchIdx, self.tokens)
        # print('batchIdx',batchIdx)
        for i, seq in tqdm(enumerate(all_hyp), mininterval=2, desc=' - (Translating)', leave=False):
            # seq2 = seq[0]
            # print('seq',seq)
            # m=src_sents[i]
            m=[]
            q=[]
            #print('past::',m)
            # print('m++',m)
            src_words, raw_words = src_tokens(src_sents[i])
            # print('src_words', src_words)
            # print('raw_words',raw_words)
            if self.opt.copy:
                is_copy_seq, copy_seq = all_is_copy[i][0], all_copy_hyp[i][0]
                preds.append(get_preds(seq, is_copy_seq=is_copy_seq, copy_seq=copy_seq, src_words=src_words))
            else:
                # preds.append(get_preds(seq))
                # print('seq',seq)
                for acc in seq:
                   # print('acc',acc)
                   m.append(tokenizer.decode(acc))
                for n in m:
                    if n not in ['<s>', '</s>']:
                       q.append(n)

                # print('current::',m)
                #assert 0==1
            preds.append(q)
            paras.append(raw_words)
        # print('+++++++paras',paras)
        # for i,token in enumerate(golds[0][0]):
        #      if token=='<pad>':
        #          index=i
        #          break
        # golds=golds[0][0][:index]
        # print('gold',golds)
        # print('gold.shape',)
        # print('preds',preds[1:-1])
        # print('preds.shape',preds.shape)
        return {'gold':golds, 'pred':preds, 'para':paras}
    
    def eval_all(self, model, validData, output_sent=False):
        all_golds, all_preds ,m,n= [], [],[],[]
        if output_sent:
            all_paras = []

        for idx in tqdm(range(len(validData)), mininterval=2, desc='   - (Translating)   ', leave=False):
            ### ========== Prepare data ========== ###
            batch = validData[idx]
            # print('batch',batch)
            inputs, gold= preprocess_batch(batch, self.opt.edge_vocab_size, sparse=self.opt.sparse, feature=self.opt.feature,
                                                               dec_feature=self.opt.dec_feature, copy=self.opt.copy, node_feature=self.opt.node_feature, 
                                                               device=self.device)
            # print('inputs',inputs)
            # copy_gold, copy_switch = copy[0], copy[1]
            # print('max_lengths',max_lengths)

            max_tgt_seq_length = inputs['max_tgt_seq_length']
            ### ========== Translate ========== ###
            # print('model',model)
            rst = self.eval_batch(model, inputs, max_tgt_seq_length, gold,
                                  # copy_gold=copy_gold, copy_switch=copy_switch,
                                  batchIdx=batch['raw-index']
                                  )
            all_golds += rst['gold']
            m.append(all_golds)
            print('\nall_golds',all_golds)
            all_preds += rst['pred']
            n.append(all_preds)
            print('inference:::',all_preds)
            # assert 0==1
            if output_sent:
                all_paras += rst['para']
        bleu = bleu_score.corpus_bleu(all_golds, all_preds)
        # bleu = bleu_score.corpus_bleu(m, n)
        print('************',output_sent)
        if output_sent:
            return bleu, (all_golds, all_preds, all_paras)
            print(00000000)
        # print(bleu)
        return bleu