#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-20$

@author: Jonathan Beaulieu-Emond
"""
import inspect

import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from PytorchTemplate.augmentations import  CutMix,MixUp,CutOut

class target_transform :
    def __init__(self,smooth : float=0,num_classes : int=10) :
        self.smooth=smooth
        self.num_classes = num_classes
    def __call__(self,i : int) :
        label = torch.zeros((self.num_classes,))+self.smooth
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



def dataset(dataset_name,num_classes,batch_size,valid_size,shuffle,num_workers,pin_memory,mean,std,root,debug,prob,label_smoothing) :






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


    dataset = getattr(datasets,dataset_name)
    cfg = {
    "root" : root,
    "train" : True,
    "download" : True,
    "transform" : train_transform,
    "target_transform" : target_transform(smooth=label_smoothing,num_classes=num_classes)

    }
    signature = inspect.signature(dataset)
    params = {key: cfg[key] for key in list(signature.parameters.keys()) if key in cfg}
    training_data = dataset(**params)
    #cfg["train"]=False
    #test_data = dataset(**cfg)
    #TODO : do data split for test depending on if parameter train in the class is available or not



    advanced_transform = transforms.Compose([
        CutMix.CutMix(p=prob[0], intensity=0.4),
        MixUp.MixUp(p=prob[1], intensity=0.4),
        #CutOut.CutOut(p=prob[2], intensity=0.4), -> already implemented in trivial augment wide
        ])
    setattr(training_data,"advanced_transform",advanced_transform)
    setattr(training_data, "normalize",normalization(mean,std,mode="train"))
    #setattr(test_data, "normalize", normalization(mean, std, mode="valid"))

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
        training_data, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return train_loader,valid_loader


if __name__ == "__main__":

    print(dir(datasets))
    num_classes = 300
    for item in dir(datasets) :
        try :
            train_loader,valid_loader = dataset(str(item),num_classes,128,0.2,True,0,False,[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010],"../../data",False,[0.5,0.5,0.5],0.1)
            image,label = next(iter(train_loader))
            print("Success for ",item)
        except Exception as e:
            print("Failed for ",item,e)