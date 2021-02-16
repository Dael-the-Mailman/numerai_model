import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def c

class BasicModel(nn.Sequential):
    def __init__(self, num_feat_input=310, feat_expansion=16):
        super.init(
            self._block(num_feat_input, feat_expansion * 8, 0.1),
            self._block(num_feat_input * 8, feat_expansion * 4, 0.1),
            self._block(num_feat_input * 4, feat_expansion * 2, 0.1),
            self._block(num_feat_input * 2, feat_expansion, 0.1),
            nn.Linear(feat_expansion, 1),
            nn.Tanh()
        )

    def _block(self, dim_in, dim_out, dropout=0.5):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.Dropout(dropout=dropout),
            nn.ReLU()
        )

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 0.01
    BATCH_SIZE = 64
    FEATURE_EXPANSION = 16

    opt = optim.Adam()

    model = BasicModel(feat_expansion=FEATURE_EXPANSION)