import cv2


def get_mask(image, model, augs, device='cuda'):
    """
    :param image: image with shame H, W, C
    :param model: segmentation torch model
    :param augs: preprocessing function for input image
    :param device: torch device
    :return: cv2 mask
    """
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = augs(image=image)['image']

    image = image.unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    res = model.predict(image)

    res = res.squeeze(dim=0).detach().cpu().numpy()
    res[res > 0.7] = 1
    res[res < 0.7] = 0
    mask = res.transpose(1, 2, 0)
    return mask

