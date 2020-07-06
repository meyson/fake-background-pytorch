import torch
import albumentations as A
import segmentation_models_pytorch as smp


def get_training_augmentation(height=480, width=640):
    train_transform = [
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.RandomCrop(height=height, width=width, p=0.5),

        A.IAAAdditiveGaussianNoise(p=0.05),
        # A.IAAPerspective(p=0.5),

        # A.OneOf(
        #     [
        #         A.CLAHE(p=1),
        #         A.RandomBrightness(p=1),
        #         A.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.IAASharpen(p=1),
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.RandomContrast(p=1),
        #         A.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height=480, width=640):
    test_transform = [
        A.Resize(height=height, width=width),
    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    x = x.transpose(2, 0, 1).astype('float32')
    return torch.from_numpy(x)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: Amentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


# processing function for inference
def get_preprocessing_fn(model_name='mobilenet_v2', height=480, width=640):
    if model_name == 'mobilenet_v2':
        preprocess_fn = smp.encoders.get_preprocessing_fn(model_name, 'imagenet')
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Lambda(image=preprocess_fn),
            A.Lambda(image=to_tensor),
        ])

    if model_name == 'org_unet':
        pass

    else:
        raise NotImplementedError()
