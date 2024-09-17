# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numpy as np 
from PIL import Image
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torch.utils.data import Dataset

class split_imagenet_dataset(Dataset):

    def __init__(self, root, txt, rate=1., transform=None):
        self.img_path = []
        self.labels = []
        self.root = root
        self.transform = transform
        self.data_rate = rate

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        num_class = 1000
        idxs = np.array(list(range(len(self.img_path)))).astype(np.long)
        targets = np.array(self.labels)

        idxList = []
        for i in range(num_class):
            idxList.append(idxs[targets == i])

        newIdxList = []
        for idxs in idxList:
            idxSampled = idxs[:int(len(idxs) * self.data_rate)]
            newIdxList += idxSampled.tolist()

        self.img_path = np.array(self.img_path)[newIdxList].tolist()
        self.labels = np.array(self.labels)[newIdxList].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label




def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("************************* BUILDING DATASET *********************************")
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if args.data_rate < 1 and is_train:
            dataset = split_imagenet_dataset(args.data_path, args.data_split, args.data_rate, transform=transform)
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
