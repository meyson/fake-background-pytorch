import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class CocoSegnentation(Dataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2017>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        augmentation (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        TODO
    """

    def __init__(self, root, ann, ann_file, augmentation=None, preprocessing=None):
        super(CocoSegnentation, self).__init__()
        from pycocotools.coco import COCO
        coco = COCO(ann_file)

        self.root = root
        self.ann = ann
        self.coco = coco
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # cats = coco.loadCats(coco.getCatIds())
        self.cats = ['person']
        self.cat_ids = coco.getCatIds(catNms=self.cats)
        self.ids = sorted(coco.getImgIds(catIds=self.cat_ids))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, mask). mask shape H,W,C
        """
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        image = cv2.imread(os.path.join(self.root, img_info['file_name']), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(
            imgIds=img_info['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )

        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((image.shape[0], image.shape[1]))
        for ann in anns:
            mask += self.coco.annToMask(ann)
        mask = (mask > 0).astype('float')
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply default preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# if __name__ == '__main__':
#     from tqdm import tqdm
#     from matplotlib import pyplot as plt
#
#     coco_root = '/mnt/xpg/datasets/fast-ai-coco/train2017/'
#     coco_ann = '/mnt/xpg/datasets/fast-ai-coco/annotations/'
#     coco_ann_file = coco_ann + 'instances_train2017.json'
#
#     # simple tests
#     dataset = CocoSegnentation(coco_root, coco_ann, coco_ann_file)
#     for i in range(5):
#         image, mask = dataset[i]
#         image = image.astype(np.uint8)
#         mask = mask.astype(np.float32)
#
#         plt.imshow(image)
#         plt.show()
#         plt.imshow(mask)
#         plt.show()

    # for img, mask in tqdm(dataset):
    #     print(img.shape)
    #     print(mask.shape)
