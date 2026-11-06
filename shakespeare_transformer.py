import math
import os
import urllib.request
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt", encoding="utf-8").read()
chars = sorted(set(text))
to_number = {c: i for i, c in enumerate(chars)}
data = torch.tensor([to_number[c] for c in text], dtype=torch.long)
