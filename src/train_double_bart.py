# 用bart
import os
import transformers
import xargs
import argparse
import math
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import cuda
import onqg.dataset.Constants as Constants
from onqg.dataset.Dataset import Dataset
from onqg.utils.model_builder import build_model
from onqg.utils.train.Loss import NLLLoss
from onqg.utils.train.Optim import Optimizer
# from onqg.utils.train.optimization import AdamW, get_linear_schedule_with_warmup
from onqg.utils.train.Train_double_bart import SupervisedTrainer
from onqg.utils.translate.Translator_double_bart import Translator
from modeling_bart_dep import BartForConditionalGeneration
# from modeling_bart import BartForConditionalGeneration
from model import CoNTGenerator
import numpy as np
# from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

class ASP_config():
    def __init__(self):
        self.K_alpha= 1
        self.V_alpha=0
        self.edge_type_number=47
        self.dependency_edge_dim=100
        self.dependency_matrix = None


class Args():
    def __init__(self):
        self.ignore_index=-100
        self.warmup=False
        self.beam_size=8
        self.diversity_pen=1.0
        self.max_length=128
        self.min_length=5
        self.no_repeat_ngram=4
        self.length_pen=2.0
        self.early_stop=True
        self.n_gram=2
        self.max_sample_num=16
        self.alpha = 0.5

def main(opt, logger):
    logger.info('My PID is {0}'.format(os.getpid()))
    logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
    logger.info(opt)

    if torch.cuda.is_available() and not opt.gpus:
        logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
    if opt.gpus:
        if opt.cuda_seed > 0:
            torch.cuda.manual_seed(opt.cuda_seed)
        cuda.set_device(opt.gpus[0])
    logger.info('My seed is {0}'.format(torch.initial_seed()))
    logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

    ###### ==================== Loading Options ==================== ######
    # if opt.checkpoint:
    #     checkpoint = torch.load(opt.checkpoint)

    ###### ==================== Loading Dataset ==================== ######
    # opt.sparse = True if opt.sparse else False
    # logger.info('Loading sequential data ......')
    # sequences = torch.load(opt.sequence_data)
    # seq_vocabularies = sequences['dict']
    # logger.info('Loading structural data ......')
    # graphs = torch.load(opt.graph_data)
    # graph_vocabularies = graphs['dict']

    ### ===== load pre-trained vocabulary ===== ###
    logger.info('Loading sequential data ......')
    sequences = torch.load(opt.sequence_data)
    # print( sequences)
    # seq_vocabularies = sequences['dict']

    
    ### ===== wrap datasets ===== ###
    logger.info('Loading Dataset objects ......')
    trainData = torch.load(opt.train_dataset, map_location=torch.device('cpu'))
    validData = torch.load(opt.valid_dataset, map_location=torch.device('cpu'))
    trainData.batchSize = validData.batchSize = opt.batch_size
    trainData.numBatches = math.ceil(len(trainData.src) / trainData.batchSize)
    validData.numBatches = math.ceil(len(validData.src) / validData.batchSize)
    # logger.info(' * number of training batches. %d' % len(trainData))
    # logger.info(' * maximum batch size. %d' % opt.batch_size)

    # del  trainData
    # del  validData

    # trainData._update_data()

    logger.info('Loading structural data ......')
    # graphs = torch.load(opt.graph_data, map_location=torch.device('cpu'))
    # print('###########################################################')
    # print('graphs',graphs)
    # graph_vocabularies = graphs['dict']
    # print('++++++++++++++++')

    # del graphs

    # opt.edge_vocab_size = graph_vocabularies['edge']['in'].size
    # opt.edge_vocab_size = graph_vocabularies['edge']['matrix_undir'].size
    # opt.node_feat_vocab = [fv.size for fv in graph_vocabularies['feature'][1:-1]] if opt.node_feature else None
    
    # logger.info(' * vocabulary size. source = %d; target = %d' % (opt.src_vocab_size, opt.tgt_vocab_size))
    logger.info(' * number of training batches. %d' % len(trainData))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    ##### =================== Prepare Model =================== #####
    device = torch.device('cuda' if opt.gpus else 'cpu')
    embedding_matrix = np.zeros((47, 100))
    asp_config = ASP_config()
    asp_config.dependency_matrix = embedding_matrix

    # trainData.device = validData.device = device
    model_args={'ASP_config':asp_config}
    args=Args()
    # print(' model_args', model_args)
    model = BartForConditionalGeneration.from_pretrained('../pre_model/bart-base',**model_args).to(device)
    # model = BartForConditionalGeneration.from_pretrained('../pre_model/bart-large').to(device)

    # nn.init.constant_(model.model.encoder.Deplayers.linear.weight, 0)
    # nn.init.constant_(model.model.encoder.Deplayers.linear.bias, 0)

    # model_name='../pre_model/bart-base'
    # model=CoNTGenerator(model_name,pad_id=1,args=args, model_args=model_args).to(device)

    logger.info('Preparing vocabularies ......')

    # opt.src_vocab_size = model.config.vocab_size
    # opt.tgt_vocab_size = model.config.vocab_size
    # opt.feat_vocab = [fv.size for fv in seq_vocabularies['feature']] if opt.feature else None

    # del checkpoint
    torch.cuda.empty_cache()
    # logger.info(' * Number of parameters to learn = %d' % parameters_cnt)
    t_total = int(len(trainData) * opt.epoch/opt.gradient_accumulation_steps)


    ##### ==================== Prepare Optimizer ==================== #####

    optimizer = Optimizer.from_opt(model, opt)

    # param = model.parameters()
    # optimizer = Adafactor(param, relative_step=False, scale_parameter=False,
    #                            lr=opt.learning_rate)
    # t_total = len(self.train_loader) * self.args.num_epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
    #                                                  num_training_steps=t_total)
    # self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
    #                                      find_unused_parameters=True)

    # cudnn.benchmark = True

    # param_optimizer = list(model.named_parameters())
    # # print('param_optimizer',param_optimizer)
    # # assert 0==1
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_grouped_paramaters = [
    #     {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)] ,
    #         "weight_decay" : 0.01
    #     },
    #     {
    #         "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0
    #     }
    # ]
    # # #
    # optimizer= AdamW(optimizer_grouped_paramaters, lr=opt.learning_rate,eps=opt.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total),
    #                                             num_training_steps=t_total)
    # ##### ==================== Prepare Loss ==================== #####
    # weight = torch.ones(opt.tgt_vocab_size)
    # weight[Constants.PAD] = 0
    # # loss = NLLLoss(opt, graph_vocabularies['feature'], model, weight, size_average=False)
    # loss = NLLLoss(opt, model, weight, size_average=False)
    # if opt.gpus:
    #     loss.cuda()

    ##### ==================== Prepare Translator ==================== #####
    # translator = Translator(opt, seq_vocabularies['tgt'], sequences['valid']['tokens'], seq_vocabularies['src'])
    # translator = Translator(opt, seq_vocabularies['tgt'], sequences['valid']['tokens'], seq_vocabularies['src'])
    # translator = Translator(opt, None, sequences['valid']['tokens'], None,args)
    translator = Translator(opt, None, sequences['valid']['tokens'], None)


    ##### ==================== Training ==================== #####
    # trainer = SupervisedTrainer(model, loss, optimizer, scheduler ,translator, logger,
    #                             opt, trainData, validData, seq_vocabularies['src'])

    trainer = SupervisedTrainer( model, optimizer,translator, logger,
                                opt, trainData, validData, None)
    del model
    # del graphs
    del trainData
    del validData
    # del seq_vocabularies['src']
    # del graph_vocabularies['feature']

    torch.cuda.empty_cache()
    trainer.train(device)


if __name__ == '__main__':
    ##### ==================== parse the options ==================== #####
    parser = argparse.ArgumentParser(description='train.py')
    xargs.add_data_options(parser)
    xargs.add_model_options(parser)
    xargs.add_train_options(parser)
    opt = parser.parse_args()

    ##### ==================== prepare the logger ==================== #####
    logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
    if opt.log_home:
        log_file_name = os.path.join(opt.log_home, log_file_name)
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    opt.copy = False

    main(opt, logger)
