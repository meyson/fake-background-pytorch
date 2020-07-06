import torch
import torch.nn as nn


class SegmentationModel(nn.Module):
    def __init__(
            self,
            pretrained: bool = True
    ):
        super().__init__()
        self.pretrained = pretrained

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
