# 用原来的bart
import os
import time
import math
import logging
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from transformers import PegasusForConditionalGeneration, BartForConditionalGeneration

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as funct

import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch


class SupervisedTrainer(object):

    def __init__(self, model, optimizer, translator, logger, opt,
                 training_data, validation_data, src_vocab):

        self.model = model
        # self.loss = loss
        self.class_loss = nn.BCELoss()
        if opt.gpus:
            self.class_loss.cuda()
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.translator = translator
        self.logger = logger
        self.opt = opt
        self.training_data = training_data
        self.validation_data = validation_data
        # self.training_data.cpu()

        # self.graph_feature_vocab = graph_feature_vocab
        # print("self.graph_feature_vocab",self.graph_feature_vocab)

        self.cntBatch = 0
        self.best_ppl, self.best_bleu, self.best_accu, self.best_kl = math.exp(100), 0, 0, 100

        self.temp = 0.05
        self.cos = nn.CosineSimilarity(dim=-1)

        self.temperature = 0.07
        self.contrast_mode = 'all'
        self.base_temperature = 0.07
        self.softmax = nn.Softmax(dim=-1)

    def cal_performance(self, loss_input, device, train_adv=True):
        # print('loss_input',loss_input)
        # gold, pred = loss_input['gold'], loss_input['pred']
        # print('pred',pred.shape)
        loss, cvl, rawl = self.loss.cal_loss(loss_input, self.model, train_adv, device)
        # print('loss',loss)

        gold, pred = loss_input['gold'], loss_input['pred']

        # print('pred',pred.shape)
        # print('gold',gold.shape)

        batch = pred.size(0)
        length = pred.size(1)
        pred = pred.contiguous().view(-1, pred.size(2))

        # print('''  ''',pred.shape)

        pred = pred.max(1)[1]
        from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('../pre_model/mbart-base')
        # print(gold.view(batch,length)[0].shape)
        # print(pred.view(batch, length)[0].shape)
        #
        # print('pred::',tokenizer.decode(pred.view(batch,length)[0].detach().cpu().numpy()))
        # print('gold::',tokenizer.decode(gold.view(batch, length)[0].detach().cpu().numpy()))
        # assert 0==1

        gold = gold.contiguous().view(-1)
        # print('gold',gold )
        # print('pred',pred)
        # print('Constants.PAD',Constants.PAD)
        # non_pad_mask = gold.ne(Constants.PAD)
        non_pad_mask = gold.ne(1)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()
        return loss, n_correct, (cvl, rawl)

    def cal_class_performance(self, loss_input, device):
        node_output, pred, gold = loss_input['node_output'], loss_input['pred'], loss_input['gold']
        # loss = self.adv(node_output, gold, device, mask=None)
        # print(loss)
        # print('node_output', node_output.shape)
        # print('pred',pred.shape)
        # print('gold',gold.shape)
        nodes, preds, golds = [], [], []
        # preds, golds =  [], []
        # for  gbatch, pbatch in zip( gold, pred):
        #     for  gw, pw in zip(gbatch, pbatch):
        #         if gw.item() != Constants.PAD:
        #             # nodes.append(nb.unsqueeze(0))
        #             golds.append(gw)
        #             preds.append(pw)

        for nbatch, gbatch, pbatch in zip(node_output, gold, pred):
            for nb, gw, pw in zip(nbatch, gbatch, pbatch):
                if gw.item() != Constants.PAD:
                    nodes.append(nb.unsqueeze(0))
                    golds.append(gw)
                    preds.append(pw)

        golds = torch.stack(golds, dim=0).to(device)
        nodes = torch.cat(nodes, dim=0).to(device)
        # print('nodes',nodes.shape)
        nodes = nodes.contiguous()
        golds = golds.eq(self.graph_feature_vocab[-1].labelToIdx[1]).float()  # TODO: magic number
        # print('+++++')
        # print('golds',golds.shape)
        # loss=self.adv(loss_input,device,mask=None)
        preds = torch.cat(preds, dim=0).to(device)
        # print('preds___',preds.shape)
        golds, preds = golds.contiguous(), preds.contiguous()

        # print('glods', golds.shape)
        # print('preds', preds.shape)
        #
        # print('golds',golds)
        # print('preds', preds)

        loss = self.adv(nodes, golds, device, mask=None)
        # print('loss',loss)

        pred_loss = self.class_loss(preds, golds) + loss

        # print('pred_loss',pred_loss)

        preds = preds.ge(0.5).view(-1).float()
        correct = preds.eq(golds)
        correct = correct.sum().item()
        return pred_loss, correct

    def get_precision_and_recall(self, loss_input, device):
        pred, gold = loss_input['pred'], loss_input['gold']
        preds, golds = [], []
        for gbatch, pbatch in zip(gold, pred):
            for gw, pw in zip(gbatch, pbatch):
                if gw.item() != Constants.PAD:
                    golds.append(gw)
                    preds.append(pw)

        golds = torch.stack(golds, dim=0).to(device)
        golds = golds.eq(self.graph_feature_vocab[-1].labelToIdx[1]).float()  # TODO: magic number

        preds = torch.cat(preds, dim=0).to(device)
        golds, preds = golds.contiguous(), preds.contiguous()

        preds = preds.ge(0.5).view(-1).float()

        sum_p, sum_r = preds.eq(1).view(-1), golds.eq(1).view(-1)
        correct = preds.eq(golds).view(-1)

        cr_p, cr_r = (correct * sum_p).sum().item(), (correct * sum_r).sum().item()

        return [cr_p, sum_p.sum().item(), cr_r, sum_r.sum().item()]

    def save_model(self, eval_num):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': self.opt,
            'step': self.cntBatch}

        if self.opt.training_mode != 'classify':

            # print('++++++',000000)
            model_name = self.opt.save_model + '.chkpt'
            # if better:
            #     torch.save(checkpoint, model_name)
            #     print('    - [Info] The checkpoint file has been updated.')
            if eval_num != 'unk' and eval_num > self.best_bleu:
                self.best_bleu = eval_num
                model_name = self.opt.save_model + '_grt_' + str(round(eval_num * 100, 5)) + '_bleu4.chkpt'
                torch.save(checkpoint, model_name)

        elif self.opt.training_mode == 'classify':
            model_name = self.opt.save_model + '_cls_' + str(round(eval_num * 100, 5)) + '_accuracy.chkpt'
            torch.save(checkpoint, model_name)
        #
        # elif better:
        #     model_name = self.opt.save_model + '_unf_' + str(round(eval_num, 5)) + '_KL.chkpt'
        #     torch.save(checkpoint, model_name)

    def eval_step(self, device, epoch):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()
        # print('_______________________')
        total_loss = {'classify': 0, 'generate': 0, 'unify': 0, 'coverage': 0, 'nll': 0}
        n_word_total, n_word_correct = 0, 0
        n_node_total, n_node_correct = 0, 0
        sample_num = 0
        precison, recall = [0, 0], [0, 0]

        with torch.no_grad():
            # for idx in tqdm(range(len(self.validation_data)), mininterval=2, desc='  - (Validation) ', leave=False):
            for idx in range(len(self.validation_data)):
                batch = self.validation_data[idx]
                # print('batch',batch)
                # assert 0==1
                inputs, golds, classification = preprocess_batch(batch, None, sparse=self.opt.sparse,
                                                                 feature=self.opt.feature,
                                                                 dec_feature=self.opt.dec_feature, copy=self.opt.copy,
                                                                 node_feature=self.opt.node_feature,
                                                                 device=device)

                # copy_gold, copy_switch = copy[0], copy[1]
                sample_num += len(golds)
                max_tgt_seq_length = inputs['max_tgt_seq_length']
                # tgt_seq_mask = tag_enc_mask[0]
                loss_input = {'classification': {}, 'generation': {}, 'unify': {}}
                ### forward ###
                graph_undir, edge_matrix_undir = inputs['graph-encoder']['graph_undir'], inputs['graph-encoder'][
                    'edge_matrix_undir']
                input_ids, attention_mask = inputs['seq-encoder']["src_seq"].to(graph_undir.device), \
                                            inputs['seq-encoder']["src_mask"].to(
                                                graph_undir.device)

                val_losses = []
                labels = inputs['decoder']['tgt_seq']
                # print('input_ids',input_ids.shape)
                # print('attention_mask',attention_mask.shape)
                # print('graph_undir',graph_undir.shape)
                # print(' edge_matrix_undir', edge_matrix_undir.shape)
                rst = self.model(input_ids.to('cuda'), attention_mask.to('cuda'), graph_undir.to('cuda'),
                                 edge_matrix_undir.to('cuda'),classification.to('cuda'),
                                 labels.to('cuda'))
                # rst = self.model(input_ids.to('cuda'), attention_mask.to('cuda'), classification.to('cuda'),
                #                  labels.to('cuda'))
                # rst = self.model(input_ids, attention_mask, classification, labels=labels)
                # loss = rst['loss']
                loss = rst[0]
                if self.cntBatch%100==0:
                    print('loss',loss)

                # print('loss',loss.shape)
                # val_losses.append(loss.item())
                # val_loss = np.mean(val_losses)

                outputs = {'generation': {}}
                outputs['generation']['loss'] = loss
                outputs['generation']['bleu'] = 'unk'

            #         rst = self.model(input_ids,attention_mask=attention_mask,graph_undir=graph_undir,edge_matrix_undir=edge_matrix_undir,
            #                     decoder_input_ids=inputs['decoder']['tgt_seq'],
            #                     decoder_attention_mask=inputs['decoder']['tgt_seq_mask'],
            #                     output_hidden_states=True)
            #
            #         # print('rst',rst)
            #         # assert 0==1
            #
            #
            #         # if self.opt.copy and self.opt.training_mode != 'classify':
            #         #     loss_input['generation']['copy_pred'], loss_input['generation']['copy_gate'] = rst['generation'][
            #         #                                                                                        'copy_pred'], \
            #         #                                                                                    rst['generation'][
            #         #                                                                                        'copy_gate']
            #         #     loss_input['generation']['copy_gold'], loss_input['generation'][
            #         #         'copy_switch'] = copy_gold, copy_switch
            #
            #         # if self.opt.coverage and self.opt.training_mode != 'classify':
            #         #     loss_input['generation']['coverage_pred'] = rst['generation']['coverage_pred']
            #
            #         if self.opt.training_mode != 'classify':
            #
            #             # m2= rst['generation']['pred']
            #             # loss_input['generation']['m2'] = rst['dec_output1']
            #             #
            #             # print('m',m2)
            #             loss_input['generation']['pred'] = rst.logits[:,:max_tgt_seq_length,:]
            #             # print("loss_input['generation']['pred'] ",loss_input['generation']['pred'])
            #             c=rst.logits[:,:-1,:]
            #             loss_input['generation']['gold'] = golds
            #             # print("golds[0]",golds[0])
            #             # print('generator check!!!!!')
            #             loss, n_correct_word, loss_package = self.cal_performance(loss_input['generation'], device, train_adv=False)
            #             coverage_loss, nll_loss = loss_package[0], loss_package[1]
            #             total_loss['generate'] += loss.item()
            #
            #             if self.opt.coverage:
            #                 total_loss['coverage'] += coverage_loss.item()
            #             total_loss['nll'] += nll_loss.item()
            #
            #         # print('golds[0]',golds[0])
            #         # non_pad_mask = golds[0].ne(Constants.PAD)
            #         non_pad_mask = golds[0].ne(1)
            #         # print('non_pad_mask',non_pad_mask)
            #         n_word = non_pad_mask.sum().item()
            #         # print('n_word',n_word)
            #
            #         # n_node = golds[1].ne(Constants.PAD).sum().item()
            #         if self.opt.training_mode != 'classify':
            #             n_word_total += n_word
            #             # print('n_correct_word',n_correct_word)
            #
            #             n_word_correct += n_correct_word
            #         # if self.opt.training_mode != 'generate':
            #         #     n_node_total += n_node
            #         #     n_node_correct += n_correct_node
            #
            # outputs = {'classification': {}, 'generation': {}, 'unify': {}}
            # if self.opt.training_mode != 'classify':
            #     outputs['generation']['loss'] = total_loss['generate'] / n_word_total
            #     outputs['generation']['correct'] = n_word_correct / n_word_total
            #     outputs['generation']['bleu'] = 'unk'
            #
            #     # print('n_word_total',n_word_total)
            #     # print('+++++++',total_loss['nll'])
            #
            #     outputs['generation']['perplexity'] = math.exp(min(total_loss['nll'] / n_word_total, 16))
            #     if self.opt.coverage:
            #         outputs['generation']['coverage'] = total_loss['coverage'] / sample_num

            # if outputs['generation']['perplexity'] <= self.opt.translate_ppl or outputs['generation'][
            #     'perplexity'] > self.best_ppl:
            # need to delete
            # self.opt.translate_steps=1

            print(self.cntBatch)
            print(self.opt.translate_steps)
            # assert 0==1
            if self.cntBatch % self.opt.translate_steps == 0:
                print('++++++__________++')
                outputs['generation']['bleu'] = self.translator.eval_all(self.model, self.validation_data)

        # # if self.opt.training_mode == 'unify':
        # outputs['unify'] = total_loss['unify']

        return outputs

    def train_epoch(self, device, epoch):
        ''' Epoch operation in training phase'''
        # if self.opt.extra_shuffle and epoch > self.opt.curriculum:
        #     self.logger.info('Shuffling...')
        #     # self.training_data.shuffle()

        self.model.train()
        total_loss = {'classify': 0, 'generate': 0, 'unify': 0, 'coverage': 0, 'nll': 0}
        n_word_total, n_word_correct = 0, 0
        n_node_total, n_node_correct = 0, 0
        report_total_loss = {'classify': 0, 'generate': 0, 'unify': 0, 'coverage': 0, 'nll': 0}
        report_n_word_total, report_n_word_correct = 0, 0
        report_n_node_total, report_n_node_correct = 0, 0
        sample_num = 0
        batch_order = torch.randperm(len(self.training_data))
        # batch_order = torch.arange(len(self.training_data))

        # need to delete

        # self.opt.valid_steps =1
        # best_loss=1e12
        n_iter, running_avg_loss = 0, 0.0
        batch_nb = len(self.training_data)
        for idx in tqdm(range(len(self.training_data)), mininterval=2, desc='  - (Training)   ', leave=False):
            batch_idx = batch_order[idx] if epoch > self.opt.curriculum else idx
            # print('+++++++++++',batch_idx)
            batch = self.training_data[batch_idx]
            # print('batch',batch)
            # assert 0==1

            # assert 0==1
            # print('encoder_input::', batch['src'].shape)
            # print('decoder_input::', batch['tgt'].shape)
            ##### ==================== prepare data ==================== #####
            inputs, golds, classification = preprocess_batch(batch, None, feature=self.opt.feature,
                                                             dec_feature=self.opt.dec_feature, copy=self.opt.copy,
                                                             node_feature=self.opt.node_feature, device=device)


            # print('classification',classification.shape)

            # inputs, golds= preprocess_batch(batch, None, feature=self.opt.feature,
            #                                                  dec_feature=self.opt.dec_feature, copy=self.opt.copy,
            #                                                  node_feature=self.opt.node_feature, device=device)

            # copy_gold, copy_switch = copy[0], copy[1]
            sample_num += len(golds)
            max_tgt_seq_length = inputs['max_tgt_seq_length']
            ##### ==================== forward ==================== #####
            self.model.zero_grad()
            self.optimizer.zero_grad()
            # edge_matrix_undir=inputs['graph-encoder']['edge_matrix_undir'].to('cuda')

            graph_undir, edge_matrix_undir = inputs['graph-encoder']['graph_undir'].to('cuda'), inputs['graph-encoder'][
                'edge_matrix_undir'].to('cuda')
            input_ids, attention_mask = inputs['seq-encoder']["src_seq"].to(device), inputs['seq-encoder'][
                "src_mask"].to(device)
            labels = inputs['decoder']['tgt_seq']
            # print('labels',labels)
            # assert 0==1
            # print('labels',labels)
            # print('input_ids',input_ids.shape)
            # print('attention_mask',attention_mask.shape)
            # print('graph_undir',graph_undir.shape)
            # print('edge_matrix_undir',edge_matrix_undir.shape)
            # assert 0==1


            rst = self.model(input_ids,attention_mask, graph_undir,edge_matrix_undir,classification=classification,labels=labels)
            # rst = self.model(input_ids, attention_mask, edge_matrix_undir, classification, labels)
            # print('classification',classification)
            # assert 0==1
            # classification=None
            # rst = self.model(input_ids, attention_mask, classification, labels=labels)
            # print('rst',rst)
            # assert 0==1

            # loss= rst['loss']
            loss = rst[0]
            # if idx % 1000 == 0:
            #     print('loss', loss)

            # print('loss',loss)
            # print('loss.shape',loss.shape)
            if len(self.opt.gpus) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if math.isnan(loss.item()) or loss.item() > 1e20:
                print('catch NaN')
                import ipdb;
                ipdb.set_trace()

            # print('loss',loss)
            # print(loss.shape)
            # self.optimizer.backward(loss)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.cntBatch += 1

            ##### ==================== evaluation ==================== #####

            # print('self.opt.valid_steps',self.opt.valid_steps)

            if self.cntBatch % self.opt.valid_steps == 0:
                ### ========== evaluation on dev ========== ###
                valid_results = self.eval_step(device, epoch)
                valid_eval = valid_results['generation']['bleu']

                # print('结果', valid_results['generation']['bleu'])
                # better = False
                # valid_eval = 0

                # if self.opt.training_mode != 'classify':
                #     # report_avg_kldiv = report_total_loss['unify'] / sample_num
                #     # report_total_loss['unify'] = 0
                #     report_avg_loss = report_total_loss['generate'] / report_n_word_total
                #     report_avg_ppl = math.exp(min(report_total_loss['nll'] / report_n_word_total, 16))
                #     report_avg_accu = report_n_word_correct / report_n_word_total
                #     # if self.opt.coverage:
                #     #     report_avg_coverage = report_total_loss['coverage'] / sample_num
                #     #     report_total_loss['coverage'] = 0
                # self.logger.info('  +  Training coverage loss: {loss:2.5f}'.format(loss=report_avg_coverage))
                #     #     self.logger.info('  +  Validation coverage loss: {loss:2.5f}'.format(
                #     #         loss=valid_results['generation']['coverage']))
                #     report_total_loss['generate'], report_total_loss['nll'] = 0, 0
                #     report_n_word_correct, report_n_word_total = 0, 0
                #
                #     # print('best_ppl',self.best_ppl)
                #     better = valid_results['generation']['perplexity'] < self.best_ppl
                #     # print('better',better)
                #
                #     if better:
                #         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #         self.best_ppl = valid_results['generation']['perplexity']
                #
                #     valid_eval = valid_results['generation']['bleu']
                #     print('结果',valid_results['generation']['bleu'])
                #
                # sample_num = 0

                ### ========== update learning rate ========== ###

                # print('better',better)
                # self.optimizer.update_learning_rate(better)
                # if self.opt.training_mode != 'classify':
                #     record_log(self.opt.logfile_train, step=self.cntBatch, loss=report_avg_loss, ppl=report_avg_ppl,
                #                accu=report_avg_accu, bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)
                #
                #     record_log(self.opt.logfile_dev, step=self.cntBatch, loss=valid_results['generation']['loss'],
                #
                #                bleu=valid_results['generation']['bleu'], bad_cnt=self.optimizer._bad_cnt,
                #                lr=self.optimizer._learning_rate)
                #
                if self.opt.save_model:
                    self.save_model(valid_eval)
                self.model.train()

        # if self.opt.training_mode == 'generate':
        #     loss_per_word = total_loss['generate'] / n_word_total
        #     # print(' loss_per_word', loss_per_word)
        #     perplexity = math.exp(min(loss_per_word, 16))
        #     accuracy = n_word_correct / n_word_total * 100
        #     m = total_loss['unify']
        #     outputs = (perplexity, accuracy,m)

        # elif self.opt.training_mode == 'classify':
        #     outputs = n_node_correct / n_node_total * 100
        # else:
        #     outputs = total_loss['unify']

    def train(self, device):
        ''' Start training '''
        for epoch_i in range(self.opt.epoch):
            self.logger.info('')
            self.logger.info(' *  [ Epoch {0} ]:   '.format(epoch_i))
            start = time.time()
            self.train_epoch(device, epoch_i + 1)
            # print(msg, end="\r")

            # if self.opt.training_mode == 'generate':
            #     self.logger.info(' *  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(ppl=results[0],accu=results[1]))
            # elif self.opt.training_mode == 'classify':
            #     self.logger.info(' *  - (Training)   accuracy: {accu: 3.3f} %'.format(accu=results))
            # else:
            #     self.logger.info(' *  - (Training)   loss: {loss: 2.5f}'.format(loss=results))
            print('                ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch_i))
