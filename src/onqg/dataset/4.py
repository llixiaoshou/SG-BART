import torch
freq = torch.Tensor(2,2,1)
freq=torch.mean(freq,-1)
print(freq.size())

# freq, idx = torch.sort(freq, 0, True)
# print(freq)
# print(idx)