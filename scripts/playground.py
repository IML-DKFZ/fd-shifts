import torch
import numpy as np

#
# x = torch.ones(2,5).cuda()
# y = torch.cuda.ByteTensor([0,1,0,1])
# print(x)

# x = np.load("/mnt/hdd2/checkpoints/checks/check_mcs/version_29/raw_output.npy")
# print(x.shape, type(x))
# softmax = x[:, :-1].reshape(7582, 10, 50)
# labels = x[:, -1]
# print(np.sum(softmax, axis=(1)))


d = {"a": 1, "b": 2, "c": 3}

print(len(d))
for ix, (k, v) in enumerate(d.items()):
    print(ix, k, v)
