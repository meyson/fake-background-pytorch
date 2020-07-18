import cv2
import numpy as np
import pyfakewebcam
import time
import argparse
import os
import functools

from ml.datasets.default_augmentations import get_test_preprocessing
from ml.inference import get_mask
from ml.models import load_from_name


parser = argparse.ArgumentParser(description='Creates fake camera changing background.')
parser.add_argument('--device', metavar='cam', dest='fake_cam', type=str, help='Virtual device (camera) to use')
parser.add_argument('--background', default='./style/backgrounds/bg1.jpg',
                    metavar='IMG', type=str, help='Background image')
parser.add_argument('--hologram-effect', dest='hologram_effect', action='store_true',
                    help='Apply hologram effect to person')
parser.add_argument('--height', metavar='N', default=480, type=int, help='Virtual camera height')
parser.add_argument('--width', metavar='N', default=640, type=int, help='Virtual camera width')
parser.add_argument('--model-name', metavar='model', dest='model_name', default='mobilenet_v2',
                    type=str, help='ML model to use')
parser.add_argument('--model-path', metavar='path', dest='model_path', default=None,
                    type=str, help='Path to saved model checkpoint')
parser.add_argument('--debug', action='store_true', help='Show debug view')


# post_process_mask from https://elder.dev/posts/open-source-virtual-background/
def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
    mask = cv2.blur(mask.astype(float), (30, 30))
    return mask


def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


# hologram_effect from https://elder.dev/posts/open-source-virtual-background/
def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if y % (bandLength + bandGap) < bandLength:
            holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out


# Decorator to measure the execution time of methods
# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
# Here I'm using time.perf_counter() since it's more accurate
def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def get_frame(cap, model, augs, background_scaled):
    _, frame = cap.read()

    mask = get_mask(frame, model, augs)

    # post-process mask and frame
    mask = post_process_mask(mask)
    if args.hologram_effect:
        frame = hologram_effect(frame)
    # composite the foreground and background
    inv_mask = 1 - mask
    for c in range(frame.shape[2]):
        frame[:, :, c] = frame[:, :, c] * mask + background_scaled[:, :, c] * inv_mask
    return frame


def main():
    global args
    args = parser.parse_args()

    # load ml models
    if args.model_path is None:
        args.model_path = os.path.join('ml', 'models', 'saved', args.model_name) + '.pth'
    model = load_from_name(model_name=args.model_name, mode='eval', path=args.model_path)
    augs = get_test_preprocessing(model_name=args.model_name)

    # load the virtual background
    background = cv2.imread(args.background)
    background_scaled = cv2.resize(background, (args.width, args.height))

    # setup access to the *real* webcam
    cap = cv2.VideoCapture(0)

    # setup the fake camera
    if args.debug:
        while True:
            frame = get_frame(cap, model, augs, background_scaled)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == 27:
                break
    else:
        fake = pyfakewebcam.FakeWebcam(args.fake_cam, args.width, args.height)
        while True:
            frame = get_frame(cap, model, augs, background_scaled)
            # fake webcam expects RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fake.schedule_frame(frame)


if __name__ == '__main__':
    main()
