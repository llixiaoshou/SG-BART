import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.functional as funct

import onqg.dataset.Constants as Constants



class Loss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0
    
    def reset(self):
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        raise NotImplementedError
    
    def cuda(self):
        self.criterion.cuda()
    
    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate. ")
        self.acc_loss.backward()

class NLLLoss(Loss):

    _NAME = "NLLLoss"

    # def __init__(self, opt, graph_feature_vocab, model, weight=None, mask=None, size_average=True, coverage_weight=0.1):
    def __init__(self, opt, graph_feature_vocab, model, weight=None, mask=None, size_average=True,coverage_weight=0.1):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask. ")
            weight[mask] = 0
        
        super(NLLLoss, self).__init__(self._NAME, nn.NLLLoss(weight=weight, size_average=size_average))

        try:
            self.opt = opt
            if opt.copy:
                self.copy_loss = nn.NLLLoss(size_average=False)
            self.coverage_weight = coverage_weight
        except:
            self.coverage_weight = coverage_weight
        
        self.KL = nn.KLDivLoss()
        # self.graph_feature_vocab = graph_feature_vocab
        self.class_loss = nn.BCELoss()
        self.tau=0.1
        if opt.gpus:
            self.class_loss.cuda()

        # self.projection = nn.Sequential(nn.Linear(hidden_size, args.hidden_size),
        #                                 nn.ReLU())
        self.tgt_vocab_size=opt.tgt_vocab_size
        # print(tgt_vocab_size)
        # self.w=model.w
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tau=0.1
        self.pos_eps=3.0
        self.neg_eps=1.0
        # self.projection = model.projection
        # self.res_class=model.res_class

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss
    
    def cal_loss(self, inputs, model, train_adv, device):
        self.opt.copy = False  # 去掉复制机制
        self.opt.coverage = False

        # print('inputs',inputs)
        # self.generator = model.generator
        pred = inputs['pred']
        # print('pred',pred.shape)
        gold = inputs['gold']
        # print('gold',gold.shape)
        # print('pred',pred.shape)

        # enc_output_mask = inputs['nodes_mask']
        # # print('enc_output_mask',enc_output_mask.shape)
        # tgt_seq_mask = inputs['tgt_seq_mask']
        # # print(' tgt_seq_mask', tgt_seq_mask.shape)
        # class_gold = inputs['class_gold']
        # enc_outputs = inputs['enc_outputs']
        # # print(' enc_outputs', enc_outputs.shape)
        # hidden_out=inputs['hidden_out']
        # # print('hidden_out',hidden_out.shape)
        # src_seq_mask=inputs['src_seq_mask']
        # # print(' src_seq_mask', src_seq_mask.shape)
        # dec_output=inputs['dec_output']

        if self.opt.copy:
            copy_pred = inputs['copy_pred']
            # print('copy_pred',copy_pred.shape)
            copy_gold = inputs['copy_gold']
            copy_gate = inputs['copy_gate']
            copy_switch = inputs['copy_switch']
        if self.opt.coverage:
            coverage_pred = inputs['coverage_pred']
        
        batch_size = gold.size(0)
        gold = gold.contiguous()
        norm = nn.Softmax(dim=1)
        device=gold.device

        pred = pred.contiguous().view(-1, pred.size(2))
        # print('_____',pred)
        # print('_____', pred.shape)
        pred = norm(pred)

        pred_prob_t = pred.contiguous().view(batch_size, -1, pred.size(1)) + 1e-8  # seq_len x batch_size x vocab_size
        # print('pred_prob_t',pred_prob_t)

        if self.opt.copy:
            # print('++++++++++++++')
            copy_pred_prob = copy_pred * copy_gate.expand_as(copy_pred) + 1e-8
            pred_prob = pred_prob_t * (1 - copy_gate).expand_as(pred_prob_t) + 1e-8
            copy_pred_prob_log = torch.log(copy_pred_prob)
            pred_prob_log1 = torch.log(pred_prob)
            # ---
            # loss_enc_tgt=self.con_loss_enc_tgt(enc_outputs, enc_output_mask,  dec_output, tgt_seq_mask, class_gold)
            copy_pred_prob_log = copy_pred_prob_log * (copy_switch.unsqueeze(2).expand_as(copy_pred_prob_log))
            pred_prob_log2 = pred_prob_log1 * ((1 - copy_switch).unsqueeze(2).expand_as(pred_prob_log1))

            pred_prob_log = pred_prob_log2.view(-1, pred_prob_log2.size(2))
            copy_pred_prob_log = copy_pred_prob_log.view(-1, copy_pred_prob_log.size(2))

            # print('pred_prob_log',pred_prob_log)

            pred_loss = self.criterion(pred_prob_log, gold.view(-1))

            # print('pred_loss',pred_loss)
            copy_loss = self.copy_loss(copy_pred_prob_log, copy_gold.contiguous().view(-1))
            
            # total_loss = loss_enc_tgt+pred_loss + copy_loss
            # print('测试++++++++++++++++++++')
            total_loss =  pred_loss + copy_loss
        else:
            pred_prob_t_log = torch.log(pred_prob_t)
            # print("1",pred_prob_t_log)
            pred_prob_t_log = pred_prob_t_log.view(-1, pred_prob_t_log.size(2))
            print('pred_prob_t_log',pred_prob_t_log.shape)
            print('gold.view(-1)',gold.view(-1).shape)
            # print('++++',pred_prob_t_log)
            # print('______', gold)
            pred_loss = self.criterion(pred_prob_t_log, gold.view(-1))
            # print('Pred_loss',pred_loss)

            total_loss = pred_loss

        # print(' total_loss', total_loss)

        # if self.opt.adv and train_adv:
        #     # print('hidden_out', hidden_out)
        #
        #
        #     proj_enc_h = self.projection(hidden_out)
        #     # print('dec_output', dec_output.shape)
        #     proj_dec_h = self.projection(dec_output)
        #
        #     avg_doc = self.avg_pool(proj_enc_h, src_seq_mask)
        #     # print('avg_doc', avg_doc.shape)
        #     avg_abs = self.avg_pool( proj_dec_h, tgt_seq_mask)
        #     # print('avg_abs', avg_abs.shape)
        #     #
        #     # cos = nn.CosineSimilarity(dim=-1)
        #     cont_crit = nn.CrossEntropyLoss()
        #     # sim_matrix = self.cos(avg_doc, avg_abs).unsqueeze(1)
        #
        #     sim_matrix = self.cos(avg_doc.unsqueeze(1),
        #                           avg_abs.unsqueeze(0))
        #     # print('sim_matrix',sim_matrix)
        #     # print('sim_matrix', sim_matrix.shape)
        #
        #     perturbed_dec = self.generate_adv(dec_output, gold.view(-1))  # [n,b,t,d] or [b,t,d]
        #     batch_size = hidden_out.size(0)
        #
        #     # print('perturbed_dec', perturbed_dec)
        #     # print('perturbed_dec', perturbed_dec.shape)
        #     proj_pert_dec_h = self.projection(perturbed_dec)
        #     # print('proj_pert_dec_h', proj_pert_dec_h)
        #     # print('proj_pert_dec_h', proj_pert_dec_h.shape)
        #     #
        #     # print('tgt_seq_mask', tgt_seq_mask)
        #     # print('tgt_seq_mask', tgt_seq_mask.shape)
        #     avg_pert = self.avg_pool(proj_pert_dec_h, tgt_seq_mask)
        #     # print('avg_doc', avg_doc)
        #     # print('avg_pert', avg_pert)
        #     adv_sim = self.cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]
        #
        #     pos_dec_hidden = self.generate_cont_adv(hidden_out, src_seq_mask,
        #                                             dec_output, tgt_seq_mask,
        #                                             pred,self.tau, self.pos_eps)
        #
        #     avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
        #                               tgt_seq_mask)
        #
        #     pos_sim = self.cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
        #     logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau
        #
        #     identity = torch.eye(batch_size).cuda()
        #     pos_sim = identity * pos_sim
        #     # print('pos_sim ', pos_sim)
        #     # print('pos_sim ', pos_sim.shape)
        #     neg_sim = sim_matrix.masked_fill(identity == 1, 0)
        #     # print('neg_sim', neg_sim)
        #     # print('neg_sim', neg_sim.shape)
        #     new_sim_matrix = pos_sim + neg_sim
        #     # print('new_sim_matrix', new_sim_matrix)
        #     # print('new_sim_matrix', new_sim_matrix.shape)
        #     new_logits = torch.cat([new_sim_matrix, adv_sim], 1)
        #     # print('new_logits', new_logits)
        #     # print('new_logits', new_logits.shape)
        #     labels = torch.arange(batch_size).cuda()
        #
        #
        #     cont_loss = cont_crit(logits, labels)
        #     # print('cont_loss', cont_loss)
        #     new_cont_loss = cont_crit(new_logits, labels)
        #     # print('new_cont_loss', new_cont_loss)
        #
        #     cont_loss = 0.5 * (cont_loss + new_cont_loss)
        #
        #     total_loss=cont_loss+total_loss

        raw_loss = total_loss
        coverage_loss = None

        if self.opt.coverage:
            coverage_pred = [cv for cv in coverage_pred]

            coverage_loss = torch.sum(torch.stack(coverage_pred, 1), 1)
            coverage_loss = torch.sum(coverage_loss, 0)
        return total_loss, coverage_loss, raw_loss

    def cal_loss_ner(self, pred, gold):
        device = gold.device
        golds = []
        for batch in gold:
            tmp_sent = torch.stack([w for w in batch if w.item() != Constants.PAD])
            golds.append(tmp_sent)
        golds = torch.cat(golds, dim=0).to(device)
        gold = golds.contiguous()
        
        pred = pred.contiguous().view(-1, pred.size(1))
        pred_prob_t_log = torch.log(pred + 1e-8)
        
        pred_loss = self.criterion(pred_prob_t_log, gold.view(-1))

        return pred_loss, gold

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2).repeat(1,1, hidden_states.size(-1)).cuda()
        # print(mask.shape)
        hidden = hidden_states.masked_fill(mask == 0, 0.0).cuda()
        # print(hidden)
        avg_hidden = torch.sum(hidden, 1) / length.cuda()
        return avg_hidden


    def con_loss_enc_tgt(self,enc_outputs, enc_output_mask, pred_prob_log, tgt_seq_mask,class_gold):

        # print('pred_prob_log',pred_prob_log.shape)
        device = class_gold.device
        enc_output_mask = enc_output_mask.unsqueeze(2).repeat(1, 1, enc_outputs.size(-1)).to(device)
        enc_outputs = enc_outputs.to(device)
        enc_sentence = torch.mul(enc_outputs, enc_output_mask).to(device)

        # cos = nn.CosineSimilarity(dim=-1)
        # print('tgt_seq_mask', tgt_seq_mask)
        # print('tgt_seq_mask', tgt_seq_mask.size())
        # print('copy_pred_prob_log', pred_prob_log.size())
        pred_tgt_sentence =self.avg_pool(pred_prob_log, tgt_seq_mask).to(device)
        # B L 512---> B 512
        pred_tgt_sentence = pred_tgt_sentence.unsqueeze(1).repeat(1, enc_sentence.size(1), 1).to(device)
        # B 512 <> B L 512
        # print("1111",enc_sentence.device)
        #
        # print('enc_sentence.size(-1)',enc_sentence.size(-1))
        # print('pred_tgt_sentence.size(-1)',pred_tgt_sentence.size(-1))
        ###############################
        #enc_sentence = nn.Linear(enc_sentence.size(-1), pred_tgt_sentence.size(-1)).cuda()(enc_sentence)
        ###############################
        # BL512 BLvocab
        # n = cos(pred_tgt_sentence, enc_sentence).unsqueeze(-1)
        # sentence <> tokens
        n = self.res_class(torch.cat([ enc_sentence, pred_tgt_sentence],dim=-1))
        # encoder_output B,L,512  dec_output B,L,512  -----> B,L,1024 -----> linear ----> BCE((B,L),gold)
        # print('n',n.shape)

        preds, golds = [], []
        for gbatch, pbatch in zip(class_gold, n):
            for gw, pw in zip(gbatch, pbatch):
                if gw.item() != Constants.PAD:
                    golds.append(gw)
                    preds.append(pw)
        # print('golds', golds)
        # print('preds', preds)
        golds = torch.stack(golds, dim=0).to(device)
        # print('golds', golds)
        golds = golds.eq(self.graph_feature_vocab[-1].labelToIdx[1]).float()  # TODO: magic number

        # print('golds++++', golds)
        # print('golds+++++', golds.shape)
        preds = torch.cat(preds, dim=0).to(device)
        # print('preds___', preds.shape)
        golds, preds = golds.contiguous(), preds.contiguous()

        # print('golds', golds)
        # print('preds', preds)
        preds = torch.ones(preds.shape).to(device)

        loss_enc_tat = self.class_loss(preds, golds)
        return loss_enc_tat


    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        # print('++++', dec_hiddens)

        dec_hiddens.requires_grad = True

        lm_logits = self.generator(dec_hiddens)



        # print('lm_logits',lm_logits)
        # print('lm_logits',lm_logits.size(-1))


        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.view(-1))

        # print('loss',loss)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()

        # print('dec_grad',dec_grad)
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        # print(' perturbed_dec', perturbed_dec.shape)
        #dec_grad.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        # print('enc_hiddens', enc_hiddens.shape)
        dec_hiddens = dec_hiddens.detach()
        # print('dec_hiddens',dec_hiddens.shape)
        lm_logits = lm_logits.detach()
        # print('lm_logits',lm_logits.shape)
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens),
                                enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens),
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)

        # print('++++',avg_enc.unsqueeze(1))
        # print('______',avg_dec.unsqueeze(0))
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        # print('logits',logits)
        # print('labels',labels)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True

        # print('perturb_dec_hidden ',perturb_dec_hidden )
        perturb_logits = self.generator(perturb_dec_hidden)
        # print('perturb_logits',perturb_logits)
        # print('perturb_logits', perturb_logits.shape)
        #
        # print('lm_logits',lm_logits)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs.view(dec_mask.size(0),dec_mask.size(1),-1)
        # print('true_probs',true_probs.shape)
        # print('dec_mask',dec_mask.shape)


        true_probs = true_probs * dec_mask.unsqueeze(-1).float()
        # print('true_probs+++',true_probs.shape)

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        # print('perturb_log_probs',perturb_log_probs.shape)
        # print('true_probs',true_probs.shape)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)
        # print('vocab_size',vocab_size)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        return perturb_dec_hidden


