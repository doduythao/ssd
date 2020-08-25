import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageFile
from utils import transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split='train', dim=(300, 300)):
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.dim = dim

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split, dim=self.dim)
        return image, boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        # Since each image may have a different num of objects, we need a collate function (passed to the DataLoader).
        # This describes how to combine these tensors of different sizes. We use lists.
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels


class COCODataset(Dataset):
    def __init__(self, list_path, dim=(300, 300), is_augment=True):
        self.dim = dim
        self.is_aug = is_augment
        with open(list_path) as f:
            self.images = f.read().splitlines()
            tmp_list = [line.replace('images', 'labels') for line in self.images]
            self.objects = [line.replace('.jpg', '.txt') for line in tmp_list]

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        w, h = image.size

        with open(self.objects[i]) as f:
            lines = f.read().splitlines()
            tmp_objs = [line.split(' ') for line in lines]
            tmp_boxes = [box[1:] for box in tmp_objs]
            labels = [int(box[0])+1 for box in tmp_objs]

        # now it's in this form: ratio c_x, c_y, b_w, b_h so convert to non-ratio
        boxes = [[float(bb[0])*w, float(bb[1])*h, float(bb[2])*w, float(bb[3])*h] for bb in tmp_boxes]
        # back to x1, y1, x2, y2.
        norm_boxes = [[int(bb[0]-bb[2]/2), int(bb[1]-bb[3]/2), int(bb[0]+bb[2]/2), int(bb[1]+bb[3]/2)] for bb in boxes]

        norm_boxes = torch.FloatTensor(norm_boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        # Apply transformations
        split = 'TRAIN' if self.is_aug else 'TEST'
        image, norm_boxes, labels = transform(image, norm_boxes, labels, split=split, dim=self.dim)
        return image, norm_boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        # Since each image may have a different num of objects, we need a collate function (passed to the DataLoader).
        # This describes how to combine these tensors of different sizes. We use lists.
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels