from transformers import MBartTokenizer
import torch
import numpy as np

tokenizer = MBartTokenizer.from_pretrained('pre_model/mbart-base')
# text='Great food but the service was unaffable !'
# word_piece_id = tokenizer(text,is_split_into_words=True, padding=True, add_special_tokens =False)
# print(word_piece_id["input_ids"])
#
# word_pieces = tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])
# print('word_pieces',word_pieces)
a = ["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?"]
