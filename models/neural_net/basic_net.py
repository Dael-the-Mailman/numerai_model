import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

DIR = "E:/datasets/numerai_data/numerai_dataset_251/numerai_training_data.csv"

def correlation(targets, predictions):
    # ranked_preds = predictions.rank(pct=True, method="first")
    return -np.corrcoef(predictions, targets)[0, 1]

class BasicModel(nn.Sequential):
    def __init__(self, num_feat_input=310, feat_expansion=16):
        super().__init__(
            self._block(num_feat_input, feat_expansion * 8),
            self._block(feat_expansion * 8, feat_expansion * 4),
            self._block(feat_expansion * 4, feat_expansion * 2),
            self._block(feat_expansion * 2, feat_expansion),
            nn.Linear(feat_expansion, 1),
            nn.ReLU()
        )

    def _block(self, dim_in, dim_out, dropout=0.1):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.Dropout(p=dropout),
            nn.PReLU()
        )

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 0.01
    BATCH_SIZE = 64
    FEATURE_EXPANSION = 16
    NUM_EPOCHS = 100

    print('Loading Data...')
    train_data = pd.read_csv(DIR)
    features = [f for f in train_data.columns if f.startswith("feature")]
    train = TensorDataset(torch.Tensor(np.array(train_data[features])), torch.Tensor(np.array(train_data["target"])))
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(features)} features")

    model = BasicModel(feat_expansion=FEATURE_EXPANSION)

    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (feat, trg) in enumerate(train_loader):
            model.zero_grad()
            opt.zero_grad()
            trg = trg.unsqueeze(1)

            # Generate output
            out = model(feat)

            # Calculate loss
            vout = out - torch.mean(out)
            vtrg = trg - torch.mean(trg)
            loss = 1 - (torch.sum(vout * vtrg) / 
                        (
                            torch.sqrt(torch.sum(vout ** 2)) * 
                            torch.sqrt(torch.sum(vtrg ** 2))
                        )
                    )
            
            # Optimize
            loss.backward()
            opt.step()
        print(f"Epoch: {epoch + 1}\tloss: {loss.item()}")
        break
    print("Success 😎")