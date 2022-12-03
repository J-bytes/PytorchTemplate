import sys

import torch
from torch.utils.data import Dataset
import os
from imageio import imread
#import re
import json
import numpy as np
import cv2 as cv
from PIL import Image
import re


class CustomImageDataset(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.transform = transform
        self.length=0
        self.files=[]
        self.annotation_files={}


        for file in os.listdir(img_dir+"/images") :
                self.files.append(f"{self.img_dir}/images/{file}")


    def __len__(self):
        return len(self.files)

    def label_transform(self,label_id): # encode one_hot
        if int(label_id)==14 :
            return torch.zeros((14))
        one_hot= torch.zeros((14))
        one_hot[int(label_id)]=1
        return one_hot.float()

    def __getitem__(self, idx):
        img_path=self.files[idx]

        patterns=img_path.split("/")[::-1]

        keyname = patterns[0]

        label_file=f"{self.img_dir}/labels/{keyname[:-3]}txt"
        if os.path.exists(label_file) :

            with open(label_file) as f:
                a=f.readlines()
                if len(a)>0 :

                    category_id = a[0].split()[0]
                else :
                    category_id=14
        else :
            category_id = 14  # the image is empty



        image = cv.imread(img_path)
        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)

        label=self.label_transform(category_id)


        return image.float(), label