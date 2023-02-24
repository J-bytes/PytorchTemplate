#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-20$

@author: Jonathan Beaulieu-Emond
"""

import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from PytorchTemplate.augmentations import  CutMix,MixUp,CutOut

class target_transform :
    def __init__(self,smooth : float=0) :
        self.smooth=smooth
    def __call__(self,i : int) :
        label = torch.zeros((10,))+self.smooth
        label[i]= 1 - self.smooth

        return label

#@torch.jit.script
class normalization :
    def __init__(self,mean : torch.Tensor,std : torch.Tensor,mode : str="train") :

        self.mode = mode
        self.normalize = transforms.Normalize(mean,std)





    def __call__(self,images):

        for ex,image in enumerate(images) :
            images[ex] = self.normalize(image)

        return images



def cifar10(batch_size,valid_size,shuffle,num_workers,pin_memory,mean,std,root,debug,prob,label_smoothing) :






    train_transform=transforms.Compose(
        [

            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),

            transforms.RandomErasing(p=0.3)

        ]
    )



    training_data = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=target_transform(smooth=label_smoothing)

    )

    valid_data = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=target_transform(smooth=0)

    )



    advanced_transform = transforms.Compose([
        CutMix.CutMix(p=prob[0], intensity=0.4),
        MixUp.MixUp(p=prob[1], intensity=0.4),
        #CutOut.CutOut(p=prob[2], intensity=0.4), -> already implemented in trivial augment wide
        ])
    setattr(training_data,"advanced_transform",advanced_transform)
    setattr(training_data, "normalize",normalization(mean,std,mode="train"))
    setattr(valid_data, "normalize", normalization(mean, std, mode="valid"))

    num_train = len(training_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        #np.random.seed(random_seed)
        np.random.shuffle(indices)

    if debug :
        train_idx, valid_idx = indices[0:1000], indices[-1000:]
    else :
        train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader,valid_loader


if __name__ == "__main__":
    cifar10()
