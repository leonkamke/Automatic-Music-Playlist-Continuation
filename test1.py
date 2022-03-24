import torch
import torch.nn as nn

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 0, 10, 8]])
    print(x.shape)
    print(x[0].argmax())
    print(x.shape)