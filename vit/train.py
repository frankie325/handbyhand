import torch
import torch.nn as nn

if __name__ == "__main__":
    cls = nn.Parameter(torch.zeros(3, 2))
    print(cls)
    print(cls.weight)
    print("Hello, World!")
