from transformers import BartTokenizer,BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('pre_model/bart-base')
# model = BartForConditionalGeneration.from_pretrained('pre_model/bart-base')



# # 原始句子
# sentence = "OpenAI is doing amazing work in natural language processing."
#
# # 使用BART分词器对句子进行标记化
# tokens = tokenizer.tokenize(sentence)
# print("Tokenized Text:", tokens)
#
# # 将标记化的结果转换为模型的输入
# input_ids = tokenizer.encode(tokens, add_special_tokens=True, return_tensors='pt')
# print("Input IDs:", input_ids)
#
# # 生成文本
# outputs = model.generate(input_ids)
# output_tokens = tokenizer.convert_ids_to_tokens(outputs[0], skip_special_tokens=True)
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Generated Text:", generated_text)
#
# # 打印分词结果
# print("Generated Tokens:", output_tokens)

text=['Arthur', "'s", 'Magazine', '(', '1844–1846', ')', 'was', 'an', 'American', 'literary', 'periodical', 'published', 'in', 'Philadelphia', 'in', 'the', '19th', 'century', '.',
'First', 'for', 'Women', 'is', 'a', 'woman', "'s", 'magazine', 'published', 'by', 'Bauer', 'Media', 'Group', 'in', 'the', 'USA', '.']
text_len=len(text)
print('text_len',text_len)

b=' '.join(text)
print('b',b)
a=tokenizer.tokenize(' '.join(text))
print(a)
# assert 0==1
word_piece_id = tokenizer(' '.join(text),is_split_into_words=True, padding=True, add_special_tokens =False)
print(word_piece_id["input_ids"])
word_pieces = tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])
print('word_pieces',word_pieces)
seq_len = len(word_pieces)
three_list=[]
star=0
for index, w in enumerate(text):
    print('w', w)
    # assert 0==1
    w_wordpice = tokenizer(w, add_special_tokens=False)
    print('w_wordpice', w_wordpice)
    # assert 0==1
    w_len = len(w_wordpice["input_ids"])
    end = star + w_len
    # three_list.append([w, index, [star, end - 1]])
    three_list.append([[star, end - 1]])
    # print('three_list',three_list)
    # assert 0==1
    star = end
# print('three_list', three_list)
# assert 0 == 1
# print(word_pieces)
# print(len(word_pieces))
print('three_list',three_list)
three_list_len=len(three_list)
print('three_list_len',three_list_len)
assert 0 == 1
# print('seq_len',seq_len)
# print('a',a)