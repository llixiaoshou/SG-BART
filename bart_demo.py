
graph_undir = torch.zeros([1,8,8]).int()
edge_matrix_undir = torch.zeros([1,8,8]).int()
rel = [[2,3,4],[1,7,0],[2,3,1],[2,1,3],[4,3,4],[2,4,4],[0,3,2]]
for i in range(0,8):
    graph_undir[0][i][0] = 1
    graph_undir[0][0][i] = 1
for r in rel:
    graph_undir[0][r[0]][r[1]] = 1
    edge_matrix_undir[0][r[0]][r[1]] = r[2]
    graph_undir[0][r[1]][r[0]] = 1
    edge_matrix_undir[0][r[1]][r[0]] = r[2]



class ASP_config():
    def __init__(self):
        self.K_alpha=1
        self.V_alpha=0
        self.edge_type_number=5
        self.dependency_edge_dim=768
        self.dependency_matrix = None

embedding_matrix = np.zeros((100, 100))

asp_config = ASP_config()
asp_config.dependency_matrix = embedding_matrix

model = BartForConditionalGeneration.from_pretrained('pre_model/bart-base', asp_config=asp_config)
tokenizer = BartTokenizer.from_pretrained("pre_model/bart-base")


input = "UN Chief Says There Is <mask> in Syria"
batch = tokenizer(input, return_tensors='pt',max_length=15,padding="max_length",truncation=True)
print('batch',batch)
assert 0==1

output_ids = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],output_hidden_states=True,graph_undir=graph_undir,edge_matrix_undir=edge_matrix_undir)
# output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(output_ids)

# 输出：['UN Chief Says There Is No War in Syria']

# from transformers import BartTokenizer, BartForConditionalGeneration
#
# tokenizer = BartTokenizer.from_pretrained("pre_model/bart-base")
#
# model = BartForConditionalGeneration.from_pretrained("pre_model/bart-base")
# from torch import cuda
# print(cuda.is_available())
