# import torch
# import torch.nn as nn
import segmentation_models_pytorch as smp
from .utils import *
from .base import *


class SMPModel(SegmentationModel):
    def __init__(self, pretrained, **kwargs):
        super().__init__(pretrained)

        if pretrained:
            kwargs['encoder_weights'] = 'imagenet'
        else:
            kwargs['encoder_weights'] = None

        # TODO add more architectures
        self.model = smp.FPN(**kwargs)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.model.predict(x)


def create_model(model_name='mobilenet_v2', pretrained=False):
    if model_name == 'mobilenet_v2':
        return SMPModel(
            pretrained=pretrained,
            encoder_name='mobilenet_v2',
            activation='sigmoid',
            classes=1,
        )

    if model_name == 'org_unet':
        from .unet import UNet
        return UNet(
            pretrained=pretrained,
            in_c=1
        )

    else:
        raise NotImplementedError()


def load_from_name(model_name='mobilenet_v2', mode='train', path=None):
    if mode == 'train':
        model = create_model(model_name, pretrained=True)
        model = model.train()

    elif mode == 'eval':
        model = create_model(model_name, pretrained=False)
        load_saved_model(model, path)
        model = model.eval()
    return model
