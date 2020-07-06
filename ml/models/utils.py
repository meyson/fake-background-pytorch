import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_saved_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
