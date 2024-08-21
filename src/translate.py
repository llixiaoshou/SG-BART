import math
import torch
from torch import cuda
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np

# from onqg.utils.translate import Translator

# from src.onqg.utils.translate.Translator_org_bart import Translator

from onqg.utils.translate.Translator_double_bart import Translator

from onqg.dataset import Dataset
from onqg.utils.model_builder import build_model
from model import CoNTGenerator
from transformers import BartForConditionalGeneration


class ASP_config():
    def __init__(self):
        self.K_alpha= 1
        self.V_alpha=0
        self.edge_type_number=45
        self.dependency_edge_dim=100
        self.dependency_matrix = None


class Args():
    def __init__(self):
        self.ignore_index=-100
        self.warmup=False
        self.beam_size=12
        self.diversity_pen=1.0
        self.max_length=128
        self.min_length=5
        self.no_repeat_ngram=4
        self.length_pen=2.0
        self.early_stop=True
        self.n_gram=2
        self.max_sample_num=16
        self.alpha = 0.5



def dump(data, filename):
    golds, preds, paras = data[0], data[1], data[2]
    with open(filename, 'w', encoding='utf-8') as f:
        for g, p, pa in zip(golds, preds, paras):
            pa = [w for w in pa if w not in ['[PAD]', '[CLS]']]
            f.write('<para>\t' + ' '.join(pa) + '\n')
            f.write('<gold>\t' + ' '.join(g[0]) + '\n')
            f.write('<pred>\t' + ' '.join(p) + '\n')
            f.write('===========================\n')


def main(opt):
    device = torch.device('cuda' if opt.cuda else 'cpu')
    # print(opt.model)
    checkpoint = torch.load(opt.model)
    # print('checkpoint',checkpoint)
    model_opt = checkpoint['settings']
    # print('______',model_opt)
    # assert 0==1
    # model_opt.gpus = opt.gpus
    # model_opt.beam_size, model_opt.batch_size = opt.beam_size, opt.batch_size
    ### Prepare Data ###
    # print('opt',opt)
    sequences = torch.load(opt.sequence_data)
    # seq_vocabularies = sequences['dict']
    # # print('seq_vocabularies',seq_vocabularies)
    validData = torch.load(opt.valid_data)
    validData.batchSize = opt.batch_size
    # print('validData.batchSize',validData.batchSize)
    validData.numBatches = math.ceil(len(validData.src) / validData.batchSize)
    # print('validData.numBatches',validData.numBatches)
    # assert 0==1
    ### Prepare Model ###
    validData.device = validData.device = device
    # model, _ = build_model(model_opt, device)

    # model = BartForConditionalGeneration.from_pretrained('../pre_model/bart-base').to(device)
    embedding_matrix = np.zeros((47, 100))
    asp_config = ASP_config()
    asp_config.dependency_matrix = embedding_matrix

    # trainData.device = validData.device = device
    model_args={'ASP_config':asp_config}
    args=Args()

    model_name='../pre_model/bart-base'
    model=CoNTGenerator(model_name,pad_id=1,args=args, model_args=model_args).to(device)
    # model= BartForConditionalGeneration.from_pretrained('../pre_model/bart-base').to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    translator = Translator(model_opt, None, sequences['valid']['tokens'], None,args)
    bleu, outputs = translator.eval_all(model, validData, output_sent=True)
    print('\nbleu-4', bleu, '\n')
    # print('outputs',outputs)
    dump(outputs, opt.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-sequence_data', required=True, help='Path to data file')
    parser.add_argument('-graph_data', required=True, help='Path to data file')
    parser.add_argument('-valid_data', required=True, help='Path to data file')
    parser.add_argument('-output', required=True, help='Path to output the predictions')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpus', default=[], nargs='+', type=int)
    opt = parser.parse_args()
    opt.cuda = True if opt.gpus else False
    if opt.cuda:
        cuda.set_device(opt.gpus[0])
    main(opt)