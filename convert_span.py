
import numpy as np
from transformers import BartTokenizer

# [[2, 3, 0, 7, 7, 7, 3, 7, 10, 8, 7, 13, 11, 13, 3]]
# sen_span = [[(1, 1), (1, 2), (1, 15), (4, 4), (5, 5), (6, 6), (4, 14), (8, 10), (9, 9), (9, 10), (11, 14), (12, 12), (12, 14), (14, 14), (15, 15)]]
# 每个位置上的子节点序号
# [[3], [], [1], [2, 7, 15], [], [], [], [4, 5, 6, 8, 11], [10], [], [9], [13], [], [12, 14], [], []]
def convert_head_to_span(all_heads):
    hpsg_lists = []
    # print('all_heads', all_heads)
    for heads in all_heads:
        n = len(heads)
        childs = [[] for i in range(n + 1)]
        left_p = [i for i in range(n + 1)]
        right_p = [i for i in range(n + 1)]

        def dfs(x):
            # print('x=', x)
            # print('childs',childs)
            # print('childs[x]', x, childs[x])
            for child in childs[x]:
                # print('child',child)
                dfs(child)

                # print(' left_p[x]', x, left_p[x])
                # print('left_p[child]', child, left_p[child])
                left_p[x] = min(left_p[x], left_p[child])

                # print('right_p[x]', x, right_p[x])
                # print('right_p[child]', child, right_p[child])
                right_p[x] = max(right_p[x], right_p[child])

                # print('left_p', left_p)
                # print('right_p', right_p)

        for i, head in enumerate(heads):
            childs[head].append(i + 1)
        dfs(0)
        # print('--------------')
        hpsg_list = []
        for i in range(1, n + 1):
            hpsg_list.append((left_p[i], right_p[i]))
        # print('hpsg_list',hpsg_list)
        hpsg_lists.append(hpsg_list)
    # print('############_________________',hpsg_lists)
    return hpsg_lists


def convert_span(example, tokenizer):
    org_sen_token = example["sen_token"]
    # print('org_sen_token ',org_sen_token )
    # assert 0==1
    # print("len(org_sen_token)", len(org_sen_token))
    all_sen_span = example["sen_span"]
    split_sen_tokens = []
    org_to_split_map = {}
    pre_tok_len = 0
    # for idx, token in enumerate(org_sen_token):
    # print('org_sen_token',org_sen_token)
    # assert 0==1
    for idx, token in enumerate(org_sen_token):
        if idx == 0:
            acc = token
        else:
            acc = ' ' + token
        # print('token',token)
        sub_tok = tokenizer.tokenize(acc)
        # print('sub_tok',sub_tok)
        # assert 0==1
        # sub_tok = [' '.join(sub_tok)]
        # print('sub_tok', sub_tok)
        # assert 0==1

        # print('len(sub_tok)',len(sub_tok))
        org_to_split_map[idx] = (pre_tok_len, len(sub_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_tok)
        split_sen_tokens.extend(sub_tok)
    # assert 0 == 1
    # print('org_to_split_map',org_to_split_map)
    # assert 0==1
    # print(split_sen_tokens)
    # print(len(split_sen_tokens))
    # assert 0==1

    # assert 0==1
    cnt_span = 0
    # print('all_sen_span',all_sen_span)
    # assert 0==1
    for sen_idx, sen_span in enumerate(all_sen_span):
        for idx, (start_ix, end_ix) in enumerate(sen_span):
            assert (start_ix <= len(sen_span) and end_ix <= len(sen_span))
            cnt_span += 1


    # print('cnt_span',cnt_span)
    # print('len(org_sen_token)',len(org_sen_token))
    assert cnt_span == len(org_sen_token)
    if cnt_span == len(org_sen_token):
        print('++++++++++++')
    else:
        print( org_sen_token)
        print(cnt_span)
        print(len(org_sen_token))
        assert 0==1
    # assert 0==1

    sub_sen_span = []
    pre_sen_len = 0
    for sen_idx, sen_span in enumerate(all_sen_span):
        # print('sen_span',sen_span)
        sen_offset = pre_sen_len
        pre_sen_len += len(sen_span)
        for idx, (start_ix, end_ix) in enumerate(sen_span):
            # print((start_ix, end_ix))
            tok_start, tok_end = org_to_split_map[sen_offset + idx]
            # sub_start_idx and sub_end_idx of children of head node
            head_spans = [
                (org_to_split_map[sen_offset + start_ix - 1][0], org_to_split_map[sen_offset + end_ix - 1][1])]
            # all other head sub_tok point to first head sub_tok
            if tok_start != tok_end:
                #                 head_spans.append((tok_start + 1, tok_end))
                sub_sen_span.append(head_spans)

                for i in range(tok_start + 1, tok_end + 1):
                    sub_sen_span.append([(i, i)])
            else:
                sub_sen_span.append(head_spans)
    # print("sub_sen_span: ", sub_sen_span)
    # input()
    # print('len(split_sen_tokens)',len(split_sen_tokens))

    # print('len(sub_sen_span)',len(sub_sen_span))
    # assert 0==1
    # assert len(sub_sen_span) == len(split_sen_tokens)
    if len(sub_sen_span) == len(split_sen_tokens):
        span_mask = np.array([['[PAD]' for i in range(len(sub_sen_span))] for j in range(len(sub_sen_span))])
    else:
        print(org_sen_token)
        print(len(sub_sen_span))
        print(len(split_sen_tokens))
        assert 0 == 1


    # making masks
    # span_mask = np.zeros((len(sub_sen_span), len(sub_sen_span)))
    #
    # span_mask= np.array([['[PAD]' for i in range(len(sub_sen_span))] for j in range(len(sub_sen_span))])


    for idx, span_list in enumerate(sub_sen_span):
        for (start_ix, end_ix) in span_list:
            span_mask[start_ix:end_ix + 1, idx] = "1"


    # record_mask records ancestor nodes for each wd
    record_mask = []
    for i in range(len(sub_sen_span)):
        i_mask = []
        for j in range(len(sub_sen_span)):
            if span_mask[i, j] == "1":
                i_mask.append(j)
        record_mask.append(i_mask)

    return split_sen_tokens, span_mask, record_mask


def test(sen_tokens, sen_heads,tokenizer):


    # print("Start test...")
    # sen_tokens = ["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as", "what", "kind", "of", "state?"]
    # print('sen_heads',sen_heads)
    # sen_tokens = ["an American literary periodical", "Arthur 's Magazine 1844–1846", "was", "published",
    #               "in Philadelphia", "in the 19th century", "is", "a woman 's magazine", "published",
    #               "by Bauer Media Group", "in the USA", "First", "for Women"]

    # sen_heads = [[2, 3, 0, 7, 7, 7, 3, 7, 10, 8, 7, 13, 11, 13, 3]]
    # sen_heads = [[0, 3, 1, 1, 4, 4, 0, 7, 8, 9, 10, 7, 12]]

    # sen_span = [[(1, 1), (1, 2), (1, 15), (4, 4), (5, 5), (6, 6), (4, 14), (8, 10), (9, 9), (9, 10), (11, 14), (12, 12), (12, 14), (14, 14), (15, 15)]]

    test_sen_span = convert_head_to_span(sen_heads)
    # print("sen_span: ", test_sen_span)
    # assert test_sen_span == sen_span
    example = {"sen_token": sen_tokens, "sen_span": test_sen_span}
    tokens, input_span_mask, record_mask = convert_span(example, tokenizer)
    # print(tokens)
    # print(input_span_mask)
    # print(record_mask)
    return tokens, input_span_mask, record_mask



# if __name__ == "__main__":
#     # sen_tokens = ["an American literary periodical", "Arthur 's Magazine 1844–1846", "was", "published",
#     #               "in Philadelphia", "in the 19th century", "is", "a woman 's magazine", "published",
#     #               "by Bauer Media Group", "in the USA", "First", "for Women"]
#     # sen_heads = [[2, 3, 0, 7, 7, 7, 3, 7, 10, 8, 7, 13, 11, 13, 3]]
#
#     sen_heads = [[0, 3, 1, 1, 4, 4, 0, 7, 8, 9, 10, 7, 12]]
#     # sen_tokens = ["The", "legislation", "allowed", "California", "to", "be", "admitted", "to", "the", "Union", "as",
#     #               "what", "kind", "of", "state?"]
#     sen_tokens = ["an American literary periodical", "Arthur 's Magazine 1844–1846", "was", "published",
#                   "in Philadelphia", "in the 19th century", "is", "a woman 's magazine", "published",
#                   "by Bauer Media Group", "in the USA", "First", "for Women"]
#
#     tokenizer = BartTokenizer.from_pretrained("../pre_model/bart-base")
#     test(sen_tokens, sen_heads, tokenizer)

