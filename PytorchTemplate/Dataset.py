#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-24$

@author: Jonathan Beaulieu-Emond
"""

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
import inspect

#@torch.jit.script



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

def Dataset(dataset : torch.utils.data.Dataset or str,num_classes : int,batch_size,valid_size,shuffle,num_workers,pin_memory,mean,std,root,debug,prob,label_smoothing,extras={}) :
    """

    :param dataset: Pytorch Dataset or string of a dataset name in torchvision.datasets
    :param num_classes: The number of classes in the dataset
    :param batch_size: The batch size. If 0 , We will try to guess the maximum batch size possible for the GPU
    :param valid_size: The ratio of image to use for validation. Suggested value : 0.1
    :param shuffle: Should the training dataset be shuffled?
    :param num_workers: The number of process to create to load the data. On windows, this value needs to be set to 0
    :param pin_memory: Leave it to True
    :param mean: The mean accross channel for the dataset
    :param std: The standard deviation accross channel for the dataset
    :param root: The location of the dataset / where to download the dataset
    :param debug: If true, set the number of samples per epoch to 1000 in order to debug the code
    :param prob: Probability for each advanced augmentation to be applied [P_CutMix,P_MixUp,P_CutOut]
    :param label_smoothing: Value of label smoothing . Suggested value : 0-0.1
    :param extras: Extra parameter needed to initialize the dataset in a dictionnary
    :return: train_loader,valid_loader
    """


    #-------- Normalization ----------------
    normalize = transforms.Normalize(mean, std)
    def normalization(images):

        for ex, image in enumerate(images):
            images[ex] = normalize(image)

        return images

    #-------- target transformation ----------------

    class target_transform:
        def __init__(self, smooth: float = 0, num_classes: int = 10):
            self.smooth = smooth
            self.num_classes = num_classes
        def __call__(self, i: int):
            label = torch.zeros((self.num_classes,)) + self.smooth
            label[i] = 1 - self.smooth

            return label
        def __call__(self, i: torch.Tensor):


            return i


    #-------- Advanced Transform ----------------
    advanced_transform = transforms.Compose([
        CutMix.CutMix(p=prob[0], intensity=0.4),
        MixUp.MixUp(p=prob[1], intensity=0.4),
        # CutOut.CutOut(p=prob[2], intensity=0.4), -> already implemented in trivial augment wide
    ])


    #-------- Dataset ----------------
    if type(dataset) == str :
        assert dataset in dir (datasets), "Dataset not found in torchvision.datasets"
        dataset = getattr(datasets,dataset)

        signature = inspect.signature(dataset.__init__)

        cfg = {
            "root" : root,
            "train" : True,
            "download" : True,
            "transform" : train_transform,
            "target_transform" : target_transform(smooth=label_smoothing, num_classes=num_classes)
        } | extras

        cfg = {k:v for k,v in cfg.items() if k in signature.parameters}
        dataset = dataset(**cfg)

    setattr(dataset, "advanced_transform", advanced_transform)
    setattr(dataset, "normalize", normalization)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)

    if debug:
        train_idx, valid_idx = indices[0:1000], indices[-1000:]
    else:
        train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )



    return train_loader,valid_loader


#@torch.jit.script
class normalization :
    def __init__(self,mean : torch.Tensor,std : torch.Tensor,mode : str="train") :

        self.mode = mode
        self.normalize = transforms.Normalize(mean,std)





    def __call__(self,images):

        for ex,image in enumerate(images) :
            images[ex] = self.normalize(image)

        return images





if __name__ == "__main__":
    train_loader,val_loader = Dataset(dataset = "CIFAR10",num_classes=10,batch_size=32, valid_size=0.1, shuffle=True, num_workers=0, pin_memory=False, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), root="../../data",debug=False,prob=[1,1],label_smoothing=0.1)

    image,label = next(iter(train_loader))

    training_data = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=train_transform,
        target_transform=None

    )

    train_loader, val_loader = Dataset(dataset=training_data, num_classes=10, batch_size=32, valid_size=0.1, shuffle=True,
                                       num_workers=0, pin_memory=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                       root="../../data", debug=False, prob=[1, 1], label_smoothing=0.1,extras = {"something" : 1})
