import torch, torch.nn as nn
class SmallCNN(nn.Module):
    def __init__(self, n_classes=7, ch=(32, 64, 128, 256)):
        super().__init__(); L, c = [], 1
        for o in ch: L += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]; c = o
        self.features, self.fc = nn.Sequential(*L), nn.Linear(c, n_classes)
    def embed(self, x): return self.features(x).mean((2, 3))            # time+freq pooled => shift-invariant embedding
    def forward(self, x): e = self.embed(x); return self.fc(e), e
